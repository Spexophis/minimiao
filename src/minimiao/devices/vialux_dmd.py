# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


"""
Control of the ViALUX V-9501 VIS through the ALP-4.3 high-speed API.
"""

import ctypes as ct
import os
import platform
from dataclasses import dataclass
from typing import Optional

import numpy as np

from minimiao import logger

ALP_DEFAULT = 0
ALP_OK = 0

# AlpDevInquire
ALP_DEVICE_NUMBER = 2000
ALP_VERSION = 2001
ALP_DEV_STATE = 2002
ALP_AVAIL_MEMORY = 2003
ALP_DDC_FPGA_TEMPERATURE = 2050
ALP_APPS_FPGA_TEMPERATURE = 2051
ALP_PCB_TEMPERATURE = 2052
ALP_DEV_DMDTYPE = 2021
ALP_DEV_DISPLAY_HEIGHT = 2057
ALP_DEV_DISPLAY_WIDTH = 2058
ALP_DEV_BUSY, ALP_DEV_READY, ALP_DEV_IDLE = 1100, 1101, 1102

# AlpDevControl
ALP_SYNCH_POLARITY = 2004
ALP_TRIGGER_EDGE = 2005
ALP_LEVEL_HIGH = 2006
ALP_LEVEL_LOW = 2007
ALP_EDGE_FALLING = 2008
ALP_EDGE_RISING = 2009
ALP_TRIGGER_TIME_OUT = 2014
ALP_TIME_OUT_ENABLE = 0
ALP_TIME_OUT_DISABLE = 1
ALP_DEV_DMD_MODE = 2064
ALP_DMD_RESUME = 0
ALP_DMD_POWER_FLOAT = 1

# AlpSeqControl
ALP_SEQ_REPEAT = 2100
ALP_FIRSTFRAME = 2101
ALP_LASTFRAME = 2102
ALP_BITNUM = 2103
ALP_BIN_MODE = 2104
ALP_BIN_NORMAL = 2105
ALP_BIN_UNINTERRUPTED = 2106
ALP_DATA_FORMAT = 2110
ALP_DATA_MSB_ALIGN = 0

# AlpSeqInquire
ALP_BITPLANES = 2200
ALP_PICNUM = 2201
ALP_PICTURE_TIME = 2203
ALP_ILLUMINATE_TIME = 2204
ALP_SYNCH_DELAY = 2205
ALP_SYNCH_PULSEWIDTH = 2206
ALP_TRIGGER_IN_DELAY = 2207
ALP_MIN_PICTURE_TIME = 2211
ALP_MIN_ILLUMINATE_TIME = 2212
ALP_MAX_PICTURE_TIME = 2213

# AlpProjControl / Inquire
ALP_PROJ_MODE = 2300
ALP_MASTER = 2301
ALP_SLAVE = 2302
ALP_PROJ_INVERSION = 2306
ALP_PROJ_UPSIDE_DOWN = 2307
ALP_PROJ_STEP = 2329
ALP_PROJ_STATE = 2400
ALP_PROJ_ACTIVE, ALP_PROJ_IDLE = 1200, 1201

ALP_ERRORS = {
    1001: "ALP not found or not ready (ALP_NOT_ONLINE)",
    1002: "ALP not in idle state (ALP_NOT_IDLE)",
    1003: "Invalid device ID (ALP_NOT_AVAILABLE)",
    1004: "Device already allocated (ALP_NOT_READY)",
    1005: "Invalid parameter (ALP_PARM_INVALID)",
    1006: "Error accessing user data (ALP_ADDR_INVALID)",
    1007: "Not enough sequence memory (ALP_MEMORY_FULL)",
    1008: "Sequence in use (ALP_SEQ_IN_USE)",
    1009: "Device halted during data transfer (ALP_HALTED)",
    1010: "Initialization error (ALP_ERROR_INIT)",
    1011: "Communication error (ALP_ERROR_COMM)",
    1012: "Device removed (ALP_DEVICE_REMOVED)",
    1013: "Onboard FPGA unconfigured (ALP_NOT_CONFIGURED)",
    1014: "Not supported by driver VlxUsbLd.sys (ALP_LOADER_VERSION)",
    1018: "Wake-up from PWR_FLOAT failed (ALP_ERROR_POWER_DOWN)",
    1019: "Driver support missing; update drivers and power-cycle",
    1020: "SDRAM initialization failed",
}

DMD_TYPES = {3: "1080p 0.95\" Type A (DLP9500)", 7: "WUXGA 0.96\"", 1: "XGA 0.7\""}


class ALPError(RuntimeError):
    def __init__(self, code: int, call: str):
        self.code = code
        super().__init__(f"{call} failed: {code} {ALP_ERRORS.get(code, 'unknown error')}")


def _find_dll(api: str = "4.3") -> str:
    if platform.system() != "Windows":
        raise OSError("ViALUX ALP API is Windows-only")
    if ct.sizeof(ct.c_void_p) != 8:
        raise OSError("Use 64-bit Python")
    names = {"4.3": "alp4395.dll", "4.2": "alpV42.dll"}
    roots = []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\ViALUX\ALP-{api}") as k:
            roots.append(winreg.QueryValueEx(k, "Path")[0])
    except OSError:
        pass
    roots.append(rf"C:\Program Files\ALP-{api}")
    for root in roots:
        p = os.path.join(root, f"ALP-{api} high-speed API", "x64", names[api])
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{names[api]} not found; pass dll_path=... explicitly")


def _load_lib(dll_path: str):
    lib = ct.CDLL(dll_path)
    L, U, pL, pU, vp = ct.c_long, ct.c_ulong, ct.POINTER(ct.c_long), ct.POINTER(ct.c_ulong), ct.c_void_p
    sig = {
        "AlpDevAlloc": [L, L, pU],
        "AlpDevHalt": [U],
        "AlpDevFree": [U],
        "AlpDevControl": [U, L, L],
        "AlpDevInquire": [U, L, pL],
        "AlpSeqAlloc": [U, L, L, pU],
        "AlpSeqFree": [U, U],
        "AlpSeqControl": [U, U, L, L],
        "AlpSeqTiming": [U, U, L, L, L, L, L],
        "AlpSeqInquire": [U, U, L, pL],
        "AlpSeqPut": [U, U, L, L, vp],
        "AlpProjStart": [U, U],
        "AlpProjStartCont": [U, U],
        "AlpProjHalt": [U],
        "AlpProjWait": [U],
        "AlpProjControl": [U, L, L],
        "AlpProjInquire": [U, L, pL],
    }
    for name, args in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = L
    return lib


@dataclass
class DMDSequence:
    id: int
    n_frames: int
    bit_depth: int
    picture_time_us: int
    illumination_time_us: int


class DMD:
    def __init__(self, dll_path: Optional[str] = None, device_num: int = 0, api: str = "4.3", logg=None):
        self.logg = logg or logger.setup_logging()
        self._lib = _load_lib(dll_path or _find_dll(api))
        self._id = ct.c_ulong(0)
        self._sequences: dict[int, DMDSequence] = {}
        self._call("AlpDevAlloc", ct.c_long(device_num), ALP_DEFAULT, ct.byref(self._id))
        self._open = True
        self.width = self.dev_inquire(ALP_DEV_DISPLAY_WIDTH)
        self.height = self.dev_inquire(ALP_DEV_DISPLAY_HEIGHT)
        self.logg.info(self.info())

    # ---- plumbing ---------------------------------------------------------
    def _call(self, name: str, *args):
        rc = getattr(self._lib, name)(*args)
        if rc != ALP_OK:
            raise ALPError(rc, name)

    def dev_inquire(self, what: int) -> int:
        v = ct.c_long(0)
        self._call("AlpDevInquire", self._id, what, ct.byref(v))
        return v.value

    def dev_control(self, what: int, value: int):
        self._call("AlpDevControl", self._id, what, value)

    def seq_inquire(self, seq: DMDSequence, what: int) -> int:
        v = ct.c_long(0)
        self._call("AlpSeqInquire", self._id, seq.id, what, ct.byref(v))
        return v.value

    def seq_control(self, seq: DMDSequence, what: int, value: int):
        self._call("AlpSeqControl", self._id, seq.id, what, value)

    def proj_inquire(self, what: int) -> int:
        v = ct.c_long(0)
        self._call("AlpProjInquire", self._id, what, ct.byref(v))
        return v.value

    def proj_control(self, what: int, value: int):
        self._call("AlpProjControl", self._id, what, value)

    # ---- info ---------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def serial(self) -> int:
        return self.dev_inquire(ALP_DEVICE_NUMBER)

    @property
    def free_memory(self) -> int:
        """Remaining on-board memory in binary frames."""
        return self.dev_inquire(ALP_AVAIL_MEMORY)

    def temperatures(self) -> dict:
        """Deg C (1 LSB = 1/256 C). Not all sensors exist on every board."""
        out = {}
        for name, key in (("ddc_fpga", ALP_DDC_FPGA_TEMPERATURE),
                          ("apps_fpga", ALP_APPS_FPGA_TEMPERATURE),
                          ("pcb", ALP_PCB_TEMPERATURE)):
            try:
                out[name] = self.dev_inquire(key) / 256
            except ALPError:
                pass
        return out

    def info(self) -> str:
        t = self.dev_inquire(ALP_DEV_DMDTYPE)
        return (f"ALP serial {self.serial}, DMD {DMD_TYPES.get(t, t)}, "
                f"{self.width}x{self.height}, free memory {self.free_memory} binary frames")

    def is_projecting(self) -> bool:
        return self.proj_inquire(ALP_PROJ_STATE) == ALP_PROJ_ACTIVE

    def _to_uint8(self, frames: np.ndarray, bit_depth: int) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim == 2:
            frames = frames[None]
        if frames.shape[1:] != self.shape:
            raise ValueError(f"frames {frames.shape[1:]} != DMD {self.shape} (rows, cols)")
        if frames.dtype == bool:
            data = frames.astype(np.uint8) * 255
        elif np.issubdtype(frames.dtype, np.integer):
            data = frames.astype(np.uint8)
            if bit_depth == 1 and data.max() <= 1:
                data = data * 255  # 0/1 input -> MSB set
        else:
            raise TypeError("frames must be bool or integer (uint8)")
        return np.ascontiguousarray(data)

    def load(self, frames: np.ndarray, bit_depth: int = 1, picture_time_us: int = 0, illumination_time_us: int = 0,
             uninterrupted: bool = True, repeat: Optional[int] = None, synch_delay_us: int = 0, synch_pulse_us: int = 0,
             trigger_in_delay_us: int = 0, chunk: int = 256) -> DMDSequence:
        """
        Upload frames (N, rows, cols) and set timing.

        bit_depth=1: bool or 0/1 or 0/255 arrays (pixel on if bit 7 set).
        bit_depth=8: uint8 grayscale (PWM, much slower).
        picture_time_us: frame period (0 = API default); illumination_time_us
        0 = maximum for the given picture time. uninterrupted=True removes the
        dark phase for binary frames (fastest; illumination = picture time).
        """
        if self.is_projecting():
            raise RuntimeError("Stop projection before loading")
        data = self._to_uint8(frames, bit_depth)
        n = data.shape[0]
        if n * bit_depth > self.free_memory:
            raise MemoryError(f"Need {n * bit_depth} binary frames, {self.free_memory} free")

        sid = ct.c_ulong(0)
        self._call("AlpSeqAlloc", self._id, bit_depth, n, ct.byref(sid))
        seq = DMDSequence(sid.value, n, bit_depth, 0, 0)
        self._sequences[seq.id] = seq
        try:
            self.seq_control(seq, ALP_DATA_FORMAT, ALP_DATA_MSB_ALIGN)
            if bit_depth == 1:
                self.seq_control(seq, ALP_BIN_MODE,
                                 ALP_BIN_UNINTERRUPTED if uninterrupted else ALP_BIN_NORMAL)
            if repeat is not None:
                self.seq_control(seq, ALP_SEQ_REPEAT, repeat)
            self.set_timing(seq, picture_time_us, illumination_time_us,
                            synch_delay_us, synch_pulse_us, trigger_in_delay_us)
            for i in range(0, n, chunk):
                block = data[i:i + chunk]
                self._call("AlpSeqPut", self._id, seq.id, i, block.shape[0],
                           block.ctypes.data_as(ct.c_void_p))
        except Exception:
            self.free(seq)
            raise
        self.logg.info(f"loaded {n} frames ({bit_depth}-bit): picture {seq.picture_time_us} us, "
                       f"illumination {seq.illumination_time_us} us, "
                       f"min picture {self.seq_inquire(seq, ALP_MIN_PICTURE_TIME)} us")
        return seq

    def set_timing(self, seq: DMDSequence, picture_time_us: int = 0,
                   illumination_time_us: int = 0, synch_delay_us: int = 0,
                   synch_pulse_us: int = 0, trigger_in_delay_us: int = 0):
        """0 = API default for any argument."""
        min_pt = self.seq_inquire(seq, ALP_MIN_PICTURE_TIME)
        if picture_time_us and picture_time_us < min_pt:
            raise ValueError(f"picture_time {picture_time_us} us < minimum {min_pt} us")
        self._call("AlpSeqTiming", self._id, seq.id, illumination_time_us, picture_time_us,
                   synch_delay_us, synch_pulse_us, trigger_in_delay_us)
        seq.picture_time_us = self.seq_inquire(seq, ALP_PICTURE_TIME)
        seq.illumination_time_us = self.seq_inquire(seq, ALP_ILLUMINATE_TIME)

    def free(self, seq: DMDSequence):
        self._call("AlpSeqFree", self._id, seq.id)
        self._sequences.pop(seq.id, None)

    def free_all(self):
        for seq in list(self._sequences.values()):
            self.free(seq)

    def set_master(self, synch_active_high: bool = True):
        """Internal timing. SYNCH OUT pulses once per frame."""
        self.proj_control(ALP_PROJ_MODE, ALP_MASTER)
        self._try(lambda: self.proj_control(ALP_PROJ_STEP, ALP_DEFAULT))
        self.dev_control(ALP_SYNCH_POLARITY, ALP_LEVEL_HIGH if synch_active_high else ALP_LEVEL_LOW)

    def set_slave(self, rising_edge: bool = True, disable_timeout: bool = True):
        """Each TRIGGER IN edge displays the next frame."""
        self.proj_control(ALP_PROJ_MODE, ALP_SLAVE)
        self.dev_control(ALP_TRIGGER_EDGE, ALP_EDGE_RISING if rising_edge else ALP_EDGE_FALLING)
        if disable_timeout:
            self._try(lambda: self.dev_control(ALP_TRIGGER_TIME_OUT, ALP_TIME_OUT_DISABLE))

    def set_step(self, condition: str = "rising"):
        """Master timing, but hold each frame until a trigger event.
        condition: 'rising', 'falling', 'high', 'low'."""
        val = {"rising": ALP_EDGE_RISING, "falling": ALP_EDGE_FALLING,
               "high": ALP_LEVEL_HIGH, "low": ALP_LEVEL_LOW}[condition]
        self.proj_control(ALP_PROJ_MODE, ALP_MASTER)
        self.proj_control(ALP_PROJ_STEP, val)

    def set_orientation(self, invert: bool = False, upside_down: bool = False):
        self.proj_control(ALP_PROJ_INVERSION, int(invert))
        self.proj_control(ALP_PROJ_UPSIDE_DOWN, int(upside_down))

    @staticmethod
    def _try(fn):
        try:
            fn()
        except ALPError:
            pass  # not supported on every API/firmware version

    def start(self, seq: DMDSequence, continuous: bool = False):
        """continuous=False: run once (or ALP_SEQ_REPEAT times) and stop."""
        self._call("AlpProjStartCont" if continuous else "AlpProjStart", self._id, seq.id)

    def wait(self):
        """Block until a non-continuous sequence has finished."""
        self._call("AlpProjWait", self._id)

    def stop(self):
        """Stop immediately; DMD returns to idle."""
        self._call("AlpDevHalt", self._id)

    def show(self, image: np.ndarray, **kw) -> DMDSequence:
        """Display one static binary image continuously."""
        self.stop()
        seq = self.load(image, **kw)
        self.set_master()
        self.start(seq, continuous=True)
        return seq

    def park(self):
        """Release mirrors to flat state (PWR_FLOAT) for long idle periods."""
        self.stop()
        self.dev_control(ALP_DEV_DMD_MODE, ALP_DMD_POWER_FLOAT)

    def wake(self):
        self.dev_control(ALP_DEV_DMD_MODE, ALP_DMD_RESUME)

    def close(self):
        if not getattr(self, "_open", False):
            return
        try:
            self.stop()
            self.free_all()
        finally:
            self._call("AlpDevFree", self._id)
            self._open = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
