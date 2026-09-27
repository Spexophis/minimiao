# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Python control for the Triggerbox (SAMD21G18A, firmware v2.0).

Protocol (spec v2.0, sec. 7): ASCII over USB-UART, 115200 8N1, CR-terminated.
    COMMAND                      ARM, RUN, STOP, RESET, STATUS, DIGSTAT3, ...
    COMMAND:type:v1;v2;...       DIG1:16:300;1300;200;100;1
Replies are "OK: ..." or "ERR N: ...".
"""

import math
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import serial
from serial.tools import list_ports

from minimiao import logger

# Hardware limits (spec v2.0)
F_CPU_HZ = 48_000_000
MAX_SYSCLK_HZ = 30_000  # MAX_SYSTEM_CLOCK_HZ
SAFE_SYSCLK_HZ = 20_000  # SAFE_SYSTEM_CLOCK_HZ (firmware warns above)
MAX_PULSE_FREQ_HZ = 20_000  # ERR 1 range
N_DIGITAL = 8
N_ANALOG = 4
N_OUTPUTS = 8
PWM_CHANNELS = (1, 2, 3, 4, 5)  # TCC1, TCC2, TC3, TC4, TC5; DIG6-8 are trigger-only per spec
MAX_MV = 5000
MAX_COM2_MSGS = 4
MAX_COM2_LEN = 10
UINT16_MAX = 65535

ERROR_CODES = {
    1: "ERR_INVALID_FREQ",
    2: "ERR_INVALID_TIME",
    3: "ERR_INVALID_PULSE",
    4: "ERR_INVALID_VOLTAGE",
    5: "ERR_INVALID_CHANNEL",
    6: "ERR_INVALID_OUTPUT",
    7: "ERR_STOP_BEFORE_START",
    8: "ERR_SYSTEM_RUNNING",
    9: "ERR_INVALID_PARAM",
    10: "ERR_INVALID_MODE",
    11: "ERR_PULSE_WIDTH_TOO_SHORT",
    12: "ERR_PULSE_WIDTH_TOO_LONG",
    13: "ERR_NOT_ARMED",
    14: "COM2TX_ERR_BUFFER_FULL",
    15: "ERR_CMD_UNKNOWN",
}


class TriggerBoxError(RuntimeError):
    def __init__(self, code: int, message: str, command: str):
        self.code = code
        self.command = command
        name = ERROR_CODES.get(code, "UNKNOWN")
        super().__init__(f"{command!r} -> ERR {code} ({name}): {message}")


# Channel definitions
@dataclass
class DigitalPulse:
    """PWM pulse train: DIGn:16:start;stop;freq;width;out"""
    channel: int
    start_ms: int
    stop_ms: int
    freq_hz: int
    width_us: int
    output: Optional[int] = None  # defaults to the channel number
    enabled: bool = True

    kind = "dig"

    @property
    def out(self) -> int:
        return self.channel if self.output is None else self.output

    def command(self) -> str:
        return (f"DIG{self.channel}:16:{self.start_ms};{self.stop_ms};"
                f"{self.freq_hz};{self.width_us};{self.out}")


@dataclass
class DigitalTrigger:
    """Level output, HIGH from start to stop: DIGTRIGn:16:start;stop;out"""
    channel: int
    start_ms: int
    stop_ms: int
    output: Optional[int] = None
    enabled: bool = True

    kind = "dig"

    @property
    def out(self) -> int:
        return self.channel if self.output is None else self.output

    def command(self) -> str:
        return f"DIGTRIG{self.channel}:16:{self.start_ms};{self.stop_ms};{self.out}"


@dataclass
class AnalogFixed:
    """Fixed voltage from start to stop, 0 V after: ANAn:16:start;stop;mV"""
    channel: int
    start_ms: int
    stop_ms: int
    mv: int
    enabled: bool = True

    kind = "ana"

    def command(self) -> str:
        return f"ANA{self.channel}:16:{self.start_ms};{self.stop_ms};{self.mv}"


@dataclass
class AnalogRamp:
    """Linear ramp: ANARAMPn:16:start;stop;start_mV;stop_mV"""
    channel: int
    start_ms: int
    stop_ms: int
    start_mv: int
    stop_mv: int
    enabled: bool = True

    kind = "ana"

    def command(self) -> str:
        return (f"ANARAMP{self.channel}:16:{self.start_ms};{self.stop_ms};"
                f"{self.start_mv};{self.stop_mv}")


@dataclass
class Com2Message:
    """Scheduled text on the secondary UART: COM2TX:s:<8-digit ms><text>"""
    time_ms: int
    text: str

    def command(self) -> str:
        return f"COM2TX:s:{self.time_ms:08d}{self.text}"


Channel = Union[DigitalPulse, DigitalTrigger, AnalogFixed, AnalogRamp]


# Sequence
@dataclass
class Sequence:
    sysclk_hz: int = 1000
    external_trigger: bool = False
    max_time_ms: Optional[int] = None
    channels: list = field(default_factory=list)
    com2: list = field(default_factory=list)

    # ---- building -------------------------------------------------------
    def add(self, *items: Union[Channel, Com2Message]) -> "Sequence":
        for it in items:
            (self.com2 if isinstance(it, Com2Message) else self.channels).append(it)
        return self

    @property
    def duration_ms(self) -> int:
        stops = [c.stop_ms for c in self.channels if c.enabled]
        return max(stops, default=0)

    @property
    def tick_us(self) -> float:
        return 1e6 / self.sysclk_hz

    def commands(self) -> list[str]:
        """Full command list, excluding RESET/ARM/RUN."""
        cmds = [f"SYSCLK:16:{self.sysclk_hz}",
                f"TRIG:8:{1 if self.external_trigger else 0}"]
        if self.max_time_ms is not None:
            cmds.append(f"MAXTIME:32:{self.max_time_ms}")
        for c in self.channels:
            cmds.append(c.command())
            prefix = "DIG" if c.kind == "dig" else "ANA"
            cmds.append(f"{prefix}{c.channel}EN:8:{int(c.enabled)}")
        cmds += [m.command() for m in self.com2]
        return cmds

    # ---- validation -----------------------------------------------------
    def validate(self, strict: bool = True) -> list[str]:
        """Check against spec limits. Raises ValueError on errors (if strict);
        returns and emits warnings for things that are allowed but risky."""
        err, warn = [], []
        f = self.sysclk_hz
        if not 1 <= f <= MAX_SYSCLK_HZ:
            err.append(f"SYSCLK {f} Hz outside 1-{MAX_SYSCLK_HZ} Hz")
        elif f > SAFE_SYSCLK_HZ:
            warn.append(f"SYSCLK {f} Hz > {SAFE_SYSCLK_HZ} Hz safe limit; "
                        "ticks may be missed with many channels active")

        seen = set()
        for c in self.channels:
            tag = f"{type(c).__name__}(ch{c.channel})"
            key = (c.kind, c.channel)
            if key in seen:
                err.append(f"{tag}: channel defined twice")
            seen.add(key)

            nmax = N_DIGITAL if c.kind == "dig" else N_ANALOG
            if not 1 <= c.channel <= nmax:
                err.append(f"{tag}: channel must be 1-{nmax}")
            if c.start_ms < 0 or c.stop_ms <= c.start_ms:
                err.append(f"{tag}: need 0 <= start < stop")
            if c.stop_ms > UINT16_MAX:
                warn.append(f"{tag}: times > {UINT16_MAX} ms sent with type code 16; "
                            "check that firmware parses them as uint32")
            if self.max_time_ms is not None and c.stop_ms > self.max_time_ms:
                err.append(f"{tag}: stop {c.stop_ms} ms > MAXTIME {self.max_time_ms} ms")
            if f < 1000 and ((c.start_ms * f) % 1000 or (c.stop_ms * f) % 1000):
                warn.append(f"{tag}: start/stop not on a tick boundary at {f} Hz")

            if c.kind == "dig" and not 1 <= c.out <= N_OUTPUTS:
                err.append(f"{tag}: output must be 1-{N_OUTPUTS}")

            if isinstance(c, DigitalPulse):
                if c.channel not in PWM_CHANNELS:
                    warn.append(f"{tag}: spec lists DIG6-8 as trigger-only (no PWM timer)")
                if not 1 <= c.freq_hz <= MAX_PULSE_FREQ_HZ:
                    err.append(f"{tag}: freq must be 1-{MAX_PULSE_FREQ_HZ} Hz")
                    continue
                period_us = 1e6 / c.freq_hz
                if not 0 < c.width_us <= UINT16_MAX:
                    err.append(f"{tag}: width must be 1-{UINT16_MAX} us")
                if c.width_us >= period_us:
                    err.append(f"{tag}: width {c.width_us} us >= period {period_us:.1f} us")
                if c.width_us < 2 * self.tick_us:
                    err.append(f"{tag}: width {c.width_us} us < 2 ticks "
                               f"({2 * self.tick_us:.1f} us at {f} Hz)")
                if f % c.freq_hz:
                    warn.append(f"{tag}: {c.freq_hz} Hz does not divide SYSCLK {f} Hz; "
                                "pulses not phase-locked to ticks")

            if isinstance(c, AnalogFixed):
                if not 0 <= c.mv <= MAX_MV:
                    err.append(f"{tag}: voltage must be 0-{MAX_MV} mV")
            if isinstance(c, AnalogRamp):
                for v in (c.start_mv, c.stop_mv):
                    if not 0 <= v <= MAX_MV:
                        err.append(f"{tag}: voltage must be 0-{MAX_MV} mV")

        # GCLK4 prescaler feasibility (spec 9.6): one integer divider 1..1024
        # must give 100..0xFFFFFF counts for SYSCLK and 100..65535 for every PWM.
        pwm = [c for c in self.channels if isinstance(c, DigitalPulse) and c.enabled
               and 1 <= c.freq_hz <= MAX_PULSE_FREQ_HZ]
        if 1 <= f <= MAX_SYSCLK_HZ and gclk4_prescaler(f, [c.freq_hz for c in pwm]) is None:
            warn.append("No common GCLK4 prescaler fits SYSCLK and all PWM frequencies; "
                        "ARM will likely return ERR 3. Spread of PWM frequencies too wide.")

        if len(self.com2) > MAX_COM2_MSGS:
            err.append(f"COM2TX: max {MAX_COM2_MSGS} messages")
        for m in self.com2:
            if not 1 <= len(m.text) <= MAX_COM2_LEN:
                err.append(f"COM2TX '{m.text}': length must be 1-{MAX_COM2_LEN}")
            if not 0 <= m.time_ms <= 99_999_999:
                err.append(f"COM2TX '{m.text}': time must fit 8 digits")
        if self.com2 and f > 1000:
            warn.append("COM2TX with other channels is only guaranteed at slow SYSCLK")

        for w in warn:
            warnings.warn(w, stacklevel=2)
        if err and strict:
            raise ValueError("Invalid sequence:\n  " + "\n  ".join(err))
        return err + warn

    # ---- preview --------------------------------------------------------
    def waveforms(self, max_samples: int = 200_000):
        """Expected outputs sampled on the tick grid (decimated if long).
        Returns t_ms, {output: bool array}, {analog_ch: mV array}."""
        n_ticks = int(math.ceil(self.duration_ms * self.sysclk_hz / 1000)) + 1
        step = max(1, n_ticks // max_samples)
        ticks = np.arange(0, n_ticks, step)
        t_ms = ticks * 1000.0 / self.sysclk_hz

        dig = {o: np.zeros(ticks.size, bool) for o in range(1, N_OUTPUTS + 1)}
        ana = {}
        for c in self.channels:
            if not c.enabled:
                continue
            active = (t_ms >= c.start_ms) & (t_ms < c.stop_ms)
            if isinstance(c, DigitalTrigger):
                dig[c.out] |= active
            elif isinstance(c, DigitalPulse):
                phase_us = ((t_ms - c.start_ms) * 1000.0) % (1e6 / c.freq_hz)
                dig[c.out] |= active & (phase_us < c.width_us)  # OR semantics (spec 5.1)
            elif isinstance(c, AnalogFixed):
                ana[c.channel] = np.where(active, c.mv, 0.0)
            elif isinstance(c, AnalogRamp):
                frac = np.clip((t_ms - c.start_ms) / (c.stop_ms - c.start_ms), 0, 1)
                ana[c.channel] = np.where(active, c.start_mv + frac * (c.stop_mv - c.start_mv), 0.0)
        return t_ms, dig, ana

    def plot(self, ax=None, show: bool = True):
        import matplotlib.pyplot as plt
        t, dig, ana = self.waveforms()
        used = [o for o, v in dig.items() if v.any()]
        rows = len(used) + (1 if ana else 0)
        fig, axes = plt.subplots(max(rows, 1), 1, sharex=True,
                                 figsize=(9, 1.0 + 0.8 * rows), squeeze=False)
        axes = axes[:, 0]
        for a, o in zip(axes, used):
            a.step(t, dig[o].astype(int), where="post", lw=0.8)
            a.set_ylabel(f"OUT{o}", rotation=0, ha="right", va="center")
            a.set_yticks([])
            a.set_ylim(-0.2, 1.2)
        if ana:
            a = axes[-1]
            for ch, v in ana.items():
                a.plot(t, v / 1000, label=f"ANA{ch}")
            a.set_ylabel("V")
            a.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (ms)")
        fig.suptitle(f"SYSCLK {self.sysclk_hz} Hz, tick {self.tick_us:.1f} us", fontsize=10)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


def gclk4_prescaler(sysclk_hz: float, pwm_freqs: list[float]) -> Optional[int]:
    """Lowest integer GCLK4 divider (1..1024) satisfying spec 9.6, or None."""
    lo = F_CPU_HZ / (0xFFFFFF * sysclk_hz)
    hi = F_CPU_HZ / (100 * sysclk_hz)
    for fp in pwm_freqs:
        lo = max(lo, F_CPU_HZ / (UINT16_MAX * fp))
        hi = min(hi, F_CPU_HZ / (100 * fp))
    p = max(1, math.ceil(lo))
    return p if p <= min(hi, 1024) else None


def suggest_sysclk(pulse_freqs: list[int], max_hz: int = SAFE_SYSCLK_HZ) -> Optional[int]:
    """Highest multiple of LCM(pulse_freqs) <= max_hz (phase-locked pulses)."""
    l = 1
    for fq in pulse_freqs:
        l = l * fq // math.gcd(l, fq)
    return (max_hz // l) * l if l <= max_hz else None


class TriggerBox:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0, quiet: float = 0.05,
                 raise_on_error: bool = True, logg=None):
        """
        timeout: max wait for the first reply byte (s)
        quiet:   reply is considered complete after this idle gap (s)
        """
        self.logg = logg or logger.setup_logging()
        self.timeout = timeout
        self.quiet = quiet
        self.raise_on_error = raise_on_error
        self.last_sequence: Optional[Sequence] = None
        self._ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0)
        time.sleep(0.1)
        self._ser.reset_input_buffer()

    # ---- context / lifecycle -------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    @staticmethod
    def list_ports() -> list[str]:
        if list_ports is None:
            return []
        return [f"{p.device}  {p.description}" for p in list_ports.comports()]

    # ---- low level -------------------------------------------------------
    def _read_reply(self, timeout: float) -> str:
        buf = bytearray()
        t0 = last = time.monotonic()
        got = False
        while True:
            n = self._ser.in_waiting
            now = time.monotonic()
            if n:
                buf += self._ser.read(n)
                last, got = now, True
            elif got and now - last >= self.quiet:
                break
            elif not got and now - t0 >= timeout:
                break
            else:
                time.sleep(0.001)
        text = buf.decode("ascii", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def send(self, cmd: str, timeout: Optional[float] = None) -> str:
        """Send one command, return the reply text. Raises TriggerBoxError on 'ERR N'."""
        cmd = cmd.strip()
        self._ser.reset_input_buffer()
        self._ser.write(cmd.encode("ascii") + b"\r")
        reply = self._read_reply(self.timeout if timeout is None else timeout)
        self.logg.info(f">> {cmd}\n   {reply.replace(chr(10), chr(10) + '   ') or '[no reply]'}")
        m = re.search(r"ERR\s*(\d+)\s*:?\s*(.*)", reply)
        if m and self.raise_on_error:
            raise TriggerBoxError(int(m.group(1)), m.group(2).strip(), cmd)
        if "warn" in reply.lower():
            warnings.warn(f"{cmd}: {reply}")
        return reply

    # ---- system commands -------------------------------------------------
    def reset(self) -> str:
        r = self.send("RESET")
        time.sleep(0.1)
        return r

    def set_sysclk(self, hz: int) -> str:
        return self.send(f"SYSCLK:16:{hz}")

    def set_trigger_mode(self, external: bool) -> str:
        return self.send(f"TRIG:8:{int(external)}")

    def set_max_time(self, ms: int) -> str:
        return self.send(f"MAXTIME:32:{ms}")

    def set_com2_baud(self, baud: int) -> str:
        # Format not given in spec v2.0; assumed to follow the uint32 convention.
        return self.send(f"COM2TXBAUD:32:{baud}")

    def arm(self) -> dict:
        """ARM and return parsed feedback (raw text in 'raw')."""
        raw = self.send("ARM", timeout=2.0)
        info = {"raw": raw}
        m = re.search(r"prescaler\s*=\s*(\d+)", raw, re.I)
        if m:
            info["gclk4_prescaler"] = int(m.group(1))
        m = re.search(r"System Clock:\s*([\d.,]+)\s*Hz", raw, re.I)
        if m:
            info["sysclk_hz"] = float(m.group(1).replace(",", "."))
        info["dig"] = {int(n): (float(fq.replace(",", ".")), rest.strip())
                       for n, fq, rest in re.findall(r"DIG(\d)\s*:\s*([\d.,]+)\s*Hz,?\s*([^\r\n]*)", raw)}
        return info

    def run(self) -> str:
        return self.send("RUN")

    def stop(self) -> str:
        return self.send("STOP")

    # ---- status ------------------------------------------------------------
    def status(self) -> tuple[str, int]:
        raw = self.send("STATUS")
        m = re.search(r"STATUS:\s*([^,\r\n]+)\s*,\s*TICK:\s*(\d+)", raw, re.I)
        return (m.group(1).strip(), int(m.group(2))) if m else (raw, -1)

    def _kv(self, raw: str) -> dict:
        return {k: v for k, v in re.findall(r"(\w+)=([^,\s]+)", raw)}

    def dig_status(self, n: int) -> dict:
        return self._kv(self.send(f"DIGSTAT{n}"))

    def ana_status(self, n: int) -> dict:
        return self._kv(self.send(f"ANASTAT{n}"))

    # ---- high level --------------------------------------------------------
    def upload(self, seq: Sequence, reset: bool = True, validate: bool = True):
        """RESET (optional) and send the full configuration. Does not ARM."""
        if validate:
            seq.validate()
        if reset:
            self.reset()
        for cmd in seq.commands():
            self.send(cmd)
        self.last_sequence = seq

    def wait_until_done(self, timeout: Optional[float] = None, poll: float = 0.1) -> tuple[str, int]:
        """Poll STATUS until the sequence has finished (auto-stop)."""
        seq = self.last_sequence
        expected = seq.duration_ms / 1000 if seq else 0.0
        if timeout is None:
            timeout = float("inf") if (seq and seq.external_trigger) else expected + 2.0
        t0, seen_running = time.monotonic(), False
        while True:
            state, tick = self.status()
            running = "RUN" in state.upper()
            seen_running |= running
            elapsed = time.monotonic() - t0
            if not running and (seen_running or elapsed > expected):
                break
            if elapsed > timeout:
                raise TimeoutError(f"Sequence not finished after {timeout:.1f} s (state {state})")
            time.sleep(poll)
        self.logg.info(f"   done: STATUS {state}, TICK {tick}")
        return state, tick

    def execute(self, seq: Sequence, wait: bool = True) -> dict:
        """RESET, configure, ARM, RUN (software trigger) and optionally wait."""
        self.upload(seq)
        info = self.arm()
        if seq.external_trigger:
            self.logg.info("   armed, waiting for external trigger on TRIG input")
        else:
            self.run()
        if wait:
            self.wait_until_done()
        return info

    def repl(self):
        print("Triggerbox terminal. Empty line or 'exit' quits.")
        raise_, self.raise_on_error = self.raise_on_error, False
        try:
            while True:
                cmd = input("tb> ").strip()
                if cmd.lower() in ("", "exit", "quit"):
                    break
                self.send(cmd)
        finally:
            self.raise_on_error = raise_


if __name__ == '__main__':
    print(TriggerBox.list_ports())

    tb = TriggerBox("COM7")

    seq = Sequence(sysclk_hz=10_000, external_trigger=False)
    seq.add(
        DigitalPulse(1, start_ms=10, stop_ms=95, freq_hz=100, width_us=200, output=1),
        DigitalTrigger(3, start_ms=0, stop_ms=101, output=3),  # HIGH from start to stop
        AnalogFixed(1, 0, 500, mv=2500),
        AnalogRamp(2, 0, 1000, start_mv=0, stop_mv=5000),
    )
    seq.validate()  # raises ValueError on errors, emits warnings for risky settings
    seq.commands()  # the command strings that will be sent
    seq.plot()  # expected waveforms, no hardware needed

    tb.upload(seq)  # RESET + config (validates first)
    info = tb.arm()  # dict: raw, gclk4_prescaler, sysclk_hz, dig{n: (Hz, text)}
    tb.run()
    tb.wait_until_done()  # polls STATUS; timeout defaults to duration + 2 s
    tb.stop()  # stop early; config kept, run() again without re-ARM

    tb.close()
