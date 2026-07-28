# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import contextlib
import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Tuple, List

import serial
from serial.tools import list_ports

from minimiao import logger

log = logging.getLogger("ell14")


class ElliptecError(Exception):
    """Base error for all Elliptec/ELL14 problems."""


class ElliptecTimeoutError(ElliptecError):
    """Raised when the device does not reply within the timeout."""


class ElliptecStatusError(ElliptecError):
    """Raised when the device reports a faulty status code."""

    def __init__(self, status: "Status", addr: int):
        self.status = status
        self.addr = addr
        super().__init__(f"device 0x{addr:X} reported faulty status: {status.name} ({status.value})")


class Status(enum.IntEnum):
    """Status codes (from the ELLx protocol "GS" reply)"""
    OK = 0
    COMM_TIMEOUT = 1
    MECH_TIMEOUT = 2
    NOT_SUPPORTED = 3
    VALUE_OUT_OF_RANGE = 4
    ISOLATED = 5
    OUT_OF_ISOLATION = 6
    INIT_ERROR = 7
    THERMAL_ERROR = 8
    BUSY = 9
    SENSOR_ERROR = 10
    MOTOR_ERROR = 11
    OUT_OF_RANGE = 12
    OVERCURRENT = 13

    @classmethod
    def _missing_(cls, value):
        obj = int.__new__(cls, value)
        obj._name_ = f"RESERVED_{value}"
        obj._value_ = value
        return obj


# Status values that are not treated as a fault by default.
_DEFAULT_VALID_STATUS = frozenset({Status.OK, Status.MECH_TIMEOUT})


@dataclass(frozen=True)
class DeviceInfo:
    serial_no: str
    model_no: int
    year: int
    fw_ver: int
    hw_ver: int
    travel: int  # travel range (deg for rotary, mm for linear)
    pulses: int  # encoder pulses over the full travel range

    @property
    def pulses_per_unit(self) -> float:
        """Encoder pulses per engineering unit (pulses/deg for a rotary stage)."""
        return self.pulses / self.travel if self.pulses > 0 and self.travel > 0 else 1.0


@dataclass(frozen=True)
class MotorInfo:
    loop_enabled: bool
    motor_enabled: bool
    current: float  # amps
    ramp_up: int
    ramp_down: int
    fw_freq: float  # Hz
    bk_freq: float  # Hz


# Expected data-field lengths per reply command (used to frame reads).
_REPLY_DATA_LEN: Dict[str, int] = {
    "IN": 30, "GS": 2, "I1": 22, "I2": 22, "I3": 22,
    "PO": 8, "HO": 8, "GJ": 8, "GV": 2, "BS": 2, "BO": 8,
}

# Commands the device may emit unprompted ("background" messages) - ignored on read.
_BACKGROUND_COMMANDS = frozenset({"BO", "BS"})

ScaleType = Union[str, float]  # "stage" | "step" | float


class ELL14:
    """
    Driver for a Thorlabs ELL14 rotation mount on a serial port.

    Parameters
    ----------
    port :
        Serial port name (e.g. ``"COM5"`` on Windows, ``"/dev/ttyUSB0"`` on Linux).
    addr :
        Device bus address, 0-15. Defaults to 0 (factory default for a single mount).
    scale :
        Position unit convention:

        * ``"stage"`` (default) - user units are engineering units (degrees), converted
          via the device's own pulses/travel calibration.
        * ``"step"``  - user units are raw encoder pulses.
        * ``float``   - user units multiplied by this factor give encoder pulses.
    baudrate, timeout :
        Serial parameters. 9600 baud is fixed by the hardware; timeout is the default
        per-command read timeout in seconds.
    auto_connect :
        If True (default), open the port and read device calibration immediately.
    """

    _PRE_MOVE_DELAY = 0.1  # s; guards against a spurious move-to-0 if commands are too close

    def __init__(self, port: str = "COM10", addr: int = 0,
                 scale: ScaleType = "stage", baudrate: int = 9600, timeout: float = 3.0, auto_connect: bool = True,
                 valid_status: Sequence[Status] = tuple(_DEFAULT_VALID_STATUS), logg=None):
        self.logg = logg or logger.setup_logging()
        self._validate_scale(scale)
        if not 0 <= addr <= 15:
            raise ValueError("addr must be between 0 and 15")

        self.port = port
        self.addr = addr
        self._scale: ScaleType = scale
        self._baudrate = baudrate
        self._timeout = timeout
        self._valid_status = frozenset(valid_status)

        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        self._info: Optional[DeviceInfo] = None
        self._pulses_per_unit: float = 1.0

        if auto_connect:
            self.connect()

    # ---- lifecycle -------------------------------------------------------------------
    def connect(self) -> "ELL14":
        """Open the serial port and read the device calibration."""
        if self._ser is not None and self._ser.is_open:
            return self
        try:
            self._ser = serial.Serial(
                self.port, baudrate=self._baudrate, timeout=self._timeout,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
        except serial.SerialException as e:
            raise ElliptecError(f"could not open serial port {self.port!r}: {e}") from e

        # Small settle + flush of any partial frames from a previous session.
        time.sleep(0.05)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        self._info = self.get_device_info()
        self._pulses_per_unit = self._info.pulses_per_unit
        log.info("connected to %s at 0x%X: %s", self.port, self.addr, self._info)
        return self

    def close(self) -> None:
        """Close the serial port."""
        if self._ser is not None:
            with contextlib.suppress(Exception):
                self._ser.close()
            self._ser = None

    def __enter__(self) -> "ELL14":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _require_open(self) -> serial.Serial:
        if not self.is_open:
            raise ElliptecError("serial port is not open; call connect() first")
        return self._ser  # type: ignore[return-value]

    # ---- static discovery ------------------------------------------------------------
    @staticmethod
    def list_ports() -> List[Tuple[str, str]]:
        """Return ``[(device, description), ...]`` for all serial ports on the system."""
        return [(p.device, p.description) for p in list_ports.comports()]

    @classmethod
    def discover(cls, port: str, addrs: Sequence[int] = range(16),
                 baudrate: int = 9600, per_addr_timeout: float = 0.4) -> List[int]:
        """
        Probe a serial port for connected Elliptec device addresses.

        Sends a status request to each candidate address and collects those that reply.
        Returns a sorted list of responding addresses. Useful before instantiating with a
        specific ``addr`` on a multi-drop bus.
        """
        found: List[int] = []
        with serial.Serial(port, baudrate=baudrate, timeout=per_addr_timeout) as ser:
            for a in addrs:
                ser.reset_input_buffer()
                ser.write(f"{a:01X}gs".encode("ascii"))
                time.sleep(0.05)
                header = ser.read(3)
                if len(header) == 3:
                    with contextlib.suppress(Exception):
                        ser.readline()  # consume the data field
                        found.append(int(header[0:1], 16))
        return sorted(set(found))

    # ---- low-level serial framing ----------------------------------------------------
    @staticmethod
    def _validate_scale(scale: ScaleType) -> None:
        if not (scale in ("stage", "step") or isinstance(scale, (int, float))):
            raise ValueError(
                "scale must be 'stage', 'step', or a numeric pulses-per-unit factor; "
                f"got {scale!r}"
            )

    def _write_command(self, ser: serial.Serial, comm: str, data: str = "") -> None:
        msg = f"{self.addr:01X}{comm}{data}".encode("ascii")
        ser.reset_input_buffer()  # discard any stale/background bytes before a fresh query
        ser.write(msg)
        log.debug("-> %r", msg)

    def _read_reply(self, ser: serial.Serial, timeout: Optional[float]) -> Tuple[str, str, int]:
        """
        Read one framed reply, skipping background messages.

        Returns ``(command, data, addr)``. Raises ElliptecTimeoutError on no/partial header.
        """
        old_to = ser.timeout
        if timeout is not None:
            ser.timeout = timeout
        try:
            deadline_guard = 64  # avoid infinite loops on a chatty bus
            while deadline_guard > 0:
                deadline_guard -= 1
                header = ser.read(3)
                if len(header) < 3:
                    raise ElliptecTimeoutError(
                        f"timed out waiting for reply header (got {len(header)} bytes)"
                    )
                try:
                    raddr = int(header[0:1], 16)
                    rcomm = header[1:3].decode("ascii")
                except (ValueError, UnicodeDecodeError) as e:
                    raise ElliptecError(f"malformed reply header {header!r}") from e

                data = ser.readline().rstrip(b"\r\n").decode("ascii", errors="replace")
                log.debug("<- addr=0x%X comm=%s data=%r", raddr, rcomm, data)

                if rcomm in _BACKGROUND_COMMANDS:
                    continue  # ignore unprompted status/position broadcasts

                expected = _REPLY_DATA_LEN.get(rcomm)
                if expected is not None and len(data) != expected:
                    raise ElliptecError(
                        f"unexpected data length for {rcomm}: expected {expected}, got {len(data)}"
                    )
                return rcomm, data, raddr
            raise ElliptecError("too many background messages; aborting read")
        finally:
            ser.timeout = old_to

    def _query(self, comm: str, data: str = "",
               expect: Optional[Sequence[str]] = None,
               timeout: Optional[float] = None) -> Tuple[str, str]:
        """
        Send a command and return ``(reply_command, reply_data)``.

        If ``expect`` is given, the reply command must be one of it.
        Thread-safe across concurrent callers on the same instance.
        """
        with self._lock:
            ser = self._require_open()
            self._write_command(ser, comm, data)
            rcomm, rdata, raddr = self._read_reply(ser, timeout)
            if raddr != self.addr:
                raise ElliptecError(
                    f"reply from unexpected address 0x{raddr:X} (expected 0x{self.addr:X})"
                )
            if expect is not None and rcomm not in expect:
                raise ElliptecError(
                    f"unexpected reply command {rcomm!r} (expected one of {list(expect)})"
                )
            return rcomm, rdata

    # ---- status handling -------------------------------------------------------------
    def _parse_status(self, data: str, check: bool = True, clear: bool = True) -> Status:
        status = Status(int(data, 16))
        if check and status not in self._valid_status:
            if clear:
                # Reading status again clears a latched fault so it isn't re-reported.
                with contextlib.suppress(ElliptecError):
                    self._query("gs", expect=["GS"])
            raise ElliptecStatusError(status, self.addr)
        return status

    def _check_reply_status(self, rcomm: str, rdata: str,
                            check: bool = True, clear: bool = True) -> Optional[Status]:
        """If the reply is a GS status word, parse (and optionally validate) it."""
        if rcomm != "GS":
            return None
        return self._parse_status(rdata, check=check, clear=clear)

    def get_status(self, check: bool = False) -> Status:
        """Read the device status word. With ``check=True``, raise on a faulty status."""
        _, data = self._query("gs", expect=["GS"], timeout=10.0)
        return self._parse_status(data, check=check, clear=False)

    # ---- unit conversion -------------------------------------------------------------
    def _to_steps(self, value: float) -> str:
        """Convert a user-unit value to an 8-hex-digit two's-complement step word."""
        if self._scale == "stage":
            steps = value * self._pulses_per_unit
        elif self._scale == "step":
            steps = value
        else:  # numeric factor
            steps = value * float(self._scale)
        steps = int(round(steps)) % 2 ** 32
        return f"{steps:08X}"

    def _from_steps(self, data: str) -> float:
        """Convert an 8-hex-digit two's-complement step word to user units."""
        raw = int(data, 16)
        raw = (raw + 2 ** 31) % 2 ** 32 - 2 ** 31  # signed 32-bit
        if self._scale == "stage":
            return raw / self._pulses_per_unit
        if self._scale == "step":
            return float(raw)
        return raw / float(self._scale)

    @property
    def scale(self) -> ScaleType:
        return self._scale

    @scale.setter
    def scale(self, value: ScaleType) -> None:
        self._validate_scale(value)
        self._scale = value

    # ---- device info -----------------------------------------------------------------
    def get_device_info(self) -> DeviceInfo:
        """Query and return the device identity/calibration block."""
        _, d = self._query("in", expect=["IN"])
        info = DeviceInfo(
            model_no=int(d[0:2], 16),
            serial_no=d[2:10],
            year=int(d[10:14]),
            fw_ver=int(d[14:16], 16),
            hw_ver=int(d[16:18], 16),
            travel=int(d[18:22], 16),
            pulses=int(d[22:30], 16),
        )
        return info

    @property
    def info(self) -> DeviceInfo:
        """Cached device info from connect (re-query with :meth:`get_device_info`)."""
        if self._info is None:
            self._info = self.get_device_info()
            self._pulses_per_unit = self._info.pulses_per_unit
        return self._info

    def get_motor_info(self, motor: int = 1) -> MotorInfo:
        """Read per-motor drive parameters (motor index 1-3; ELL14 uses 1)."""
        if motor not in (1, 2, 3):
            raise ValueError("motor must be 1, 2, or 3")
        comm = f"i{motor}"
        rcomm, d = self._query(comm, expect=[comm.upper(), "GS"])
        self._check_reply_status(rcomm, d)
        return MotorInfo(
            loop_enabled=bool(int(d[0])),
            motor_enabled=bool(int(d[1])),
            current=int(d[2:6], 16) / 1866.0,
            ramp_up=int(d[6:10], 16),
            ramp_down=int(d[10:14], 16),
            fw_freq=14_740_000 / int(d[14:18], 16),
            bk_freq=14_740_000 / int(d[18:22], 16),
        )

    # ---- homing ----------------------------------------------------------------------
    def home(self, direction: str = "cw", timeout: float = 30.0) -> bool:
        """
        Home the mount (synchronous). ``direction`` is ``"cw"`` or ``"ccw"`` for a rotary
        stage. Returns True if a position reply confirmed completion.
        """
        dir_code = {"cw": 0, "ccw": 1}.get(direction)
        if dir_code is None:
            raise ValueError("direction must be 'cw' or 'ccw'")
        rcomm, rdata = self._query("ho", data=f"{dir_code:01X}",
                                   expect=["GS", "PO"], timeout=timeout)
        self._check_reply_status(rcomm, rdata)
        return rcomm == "PO"

    def get_home_offset(self) -> float:
        _, d = self._query("go", expect=["HO"])
        return self._from_steps(d)

    def set_home_offset(self, offset: float) -> float:
        """Set the homing offset (Thorlabs advises against changing this)."""
        rcomm, rdata = self._query("so", data=self._to_steps(offset), expect=["GS"])
        self._check_reply_status(rcomm, rdata)
        return self.get_home_offset()

    # ---- velocity & frequency --------------------------------------------------------
    def get_velocity(self) -> int:
        """Velocity as a percentage of maximum (0-100)."""
        _, d = self._query("gv", expect=["GV"])
        return int(d, 16)

    def set_velocity(self, percent: int) -> int:
        if not 0 <= percent <= 100:
            raise ValueError("velocity percent must be between 0 and 100")
        rcomm, rdata = self._query("sv", data=f"{int(percent):02X}", expect=["GS"])
        self._check_reply_status(rcomm, rdata)
        return self.get_velocity()

    def get_frequency(self, motor: int = 1) -> Tuple[float, float]:
        """Return ``(forward_freq, backward_freq)`` in Hz for the given motor."""
        mi = self.get_motor_info(motor)
        return mi.fw_freq, mi.bk_freq

    def search_frequency(self, motor: int = 1) -> Tuple[float, float]:
        """Run the automated resonant-frequency search (position may shift slightly)."""
        if motor not in (1, 2, 3):
            raise ValueError("motor must be 1, 2, or 3")
        comm = f"s{motor}"
        rcomm, rdata = self._query(comm, expect=["GS"], timeout=10.0)
        self._check_reply_status(rcomm, rdata)
        return self.get_frequency(motor)

    def save_parameters(self) -> Status:
        """Persist current parameters (e.g. frequencies) to non-volatile memory."""
        rcomm, rdata = self._query("us", expect=["GS"])
        return self._parse_status(rdata) if rcomm == "GS" else Status.OK

    # ---- position & motion -----------------------------------------------------------
    def get_position(self) -> float:
        """Current absolute position in user units (degrees by default)."""
        _, d = self._query("gp", expect=["PO"])
        return self._from_steps(d)

    def move_to(self, position: float, timeout: float = 30.0) -> bool:
        """
        Move to an absolute position (synchronous). Returns True if a position reply
        confirmed the move completed.
        """
        time.sleep(self._PRE_MOVE_DELAY)  # avoid a spurious move-to-0 on back-to-back moves
        rcomm, rdata = self._query("ma", data=self._to_steps(position),
                                   expect=["GS", "PO"], timeout=timeout)
        self._check_reply_status(rcomm, rdata)
        return rcomm == "PO"

    def move_by(self, distance: float, timeout: float = 30.0) -> bool:
        """Move by a relative distance (synchronous). Returns True on confirmed completion."""
        time.sleep(self._PRE_MOVE_DELAY)
        rcomm, rdata = self._query("mr", data=self._to_steps(distance),
                                   expect=["GS", "PO"], timeout=timeout)
        self._check_reply_status(rcomm, rdata)
        return rcomm == "PO"

    def move_to_wrapped(self, angle_deg: float, timeout: float = 30.0) -> bool:
        """
        Convenience for rotary stages: move to ``angle_deg`` modulo the full travel range,
        so e.g. 370 deg maps to 10 deg. Only meaningful with ``scale="stage"``.
        """
        travel = self.info.travel
        return self.move_to(angle_deg % travel, timeout=timeout)

    def jog(self, forward: bool = True, timeout: float = 30.0) -> bool:
        """Jog one configured step forward (``fw``) or backward (``bw``)."""
        comm = "fw" if forward else "bw"
        rcomm, rdata = self._query(comm, expect=["GS", "PO"], timeout=timeout)
        self._check_reply_status(rcomm, rdata)
        return rcomm == "PO"

    def get_jog_step(self) -> float:
        _, d = self._query("gj", expect=["GJ"])
        return self._from_steps(d)

    def set_jog_step(self, step: float) -> float:
        rcomm, rdata = self._query("sj", data=self._to_steps(step), expect=["GS"])
        self._check_reply_status(rcomm, rdata)
        return self.get_jog_step()

    def stop(self) -> None:
        """
        Request a stop of the current motion. Elliptec has no universal stop opcode; a
        fresh status query interrupts blocking waits on this side and returns control.
        """
        with contextlib.suppress(ElliptecError):
            self.get_status()

    # ---- address management ----------------------------------------------------------
    def change_address(self, new_addr: int) -> Status:
        """Change this device's bus address (0-15) and retarget the driver to it."""
        if not 0 <= new_addr <= 15:
            raise ValueError("new_addr must be between 0 and 15")
        with self._lock:
            ser = self._require_open()
            self._write_command(ser, "ca", f"{new_addr:01X}")
            # Reply comes from the NEW address.
            rcomm, rdata, raddr = self._read_reply(ser, self._timeout)
            if raddr != new_addr:
                raise ElliptecError(
                    f"address change reply from 0x{raddr:X}, expected 0x{new_addr:X}"
                )
            status = self._parse_status(rdata) if rcomm == "GS" else Status.OK
            self.addr = new_addr
            return status

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        model = self._info.model_no if self._info else "?"
        return f"<ELL14 port={self.port!r} addr=0x{self.addr:X} model={model} [{state}]>"
