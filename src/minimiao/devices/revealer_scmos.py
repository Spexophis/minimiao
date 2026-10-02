# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.
"""
Agiledevice (中科视界) Revealer / Gloria sCMOS camera, via the SCSDK Python wrapper.

Drop-in replacement for andor_emccd.EMCCDCamera / hamamatsu_scmos.HamamatsuCamera: same attributes and the same
methods the executor calls, so it can be selected in DeviceManager.

The SCSDK python wrapper (scsdk.py / SCDefines.py, shipped in <SDK>/samples/python/scsdk) is imported lazily, so this
module can be imported on a machine without the SDK; the constructor raises and DeviceManager falls back to MockCamera.
"""

import importlib
import os
import sys
import time
from ctypes import POINTER, byref, c_double, c_int64, c_ubyte, c_uint16, c_uint64, c_void_p, cast

import numpy as np

from minimiao import run_threads, logger

# scsdk.py / SCDefines.py live in <SDK home>/samples/python/scsdk. scsdk.py also loads <SDK home>/bin/scsdk.dll, so it
# needs the Revealer_Scientific_Camera_SDK_HOME environment variable (set by the SDK installer).
# REVEALER_SCSDK_PATH can point at the wrapper folder explicitly.
SDK_HOME_ENV = "Revealer_Scientific_Camera_SDK_HOME"
SCSDK_PATHS = [
    os.environ.get("REVEALER_SCSDK_PATH", ""),
    os.path.join(os.environ.get(SDK_HOME_ENV, ""), "samples", "python", "scsdk"),
]

# ExposureTime is a GenICam float. GenICam cameras take microseconds; the manual does not state the unit, so the
# value is read back and logged in set_exposure_time(). Change to 1e3 / 1.0 if your camera reports ms / s.
EXPOSURE_TIME_SCALE = 1e6  # feature units per second

LIVE_BUFFER_SIZE = 128
MIN_BUFFER_COUNT = 16  # SDK frame buffers (each is a full frame, 8 MB at 2048x2048x16 bit)
MAX_BUFFER_COUNT = 64
COOLING_TARGET = -10  # deg C, clamped to the range the camera accepts
GRAB_TIMEOUT_MS = 10  # first GetFrame of a polling round
DRAIN_TIMEOUT_MS = 1  # further GetFrame calls of the same round
MAX_FRAMES_PER_POLL = 64

# Gloria6504 attribute table, enumeration values
Binning_Mode = {1: 0, 2: 1, 4: 2}  # binning factor -> BinningMode enum (OneByOne, TwoByTwo, FourByFour)

Readout_Mode = {1: "bit12_STD_High (high speed, high gain)",
                2: "bit16_HDR (high dynamic range)"}  # ReadoutMode enum, value 0 is not listed in the table

Trigger_Mode = {0: "Off",
                1: "External_Edge_Trigger",
                2: "External_Start_Trigger",
                3: "External_Level_Trigger",
                4: "Synchronous_Readout",
                5: "Software_Trigger"}  # TriggerInType enum

Trigger_Activation = {0: "RisingEdge", 1: "FallingEdge", 2: "LevelHigh", 3: "LevelLow"}

Shutter_Mode = {0: "Rolling", 1: "GlobalReset", 2: "ProgramableMode"}


class RevealerCamera:
    class CameraSettings:
        def __init__(self):
            self.temperature = None
            self.gain = 0  # sCMOS: no EM gain, kept so the GUI/executor attribute exists
            self.t_clean = 0.001
            self.t_readout = 0.01
            self.t_exposure = 0.0403
            self.t_accumulate = None
            self.t_kinetic = 0.05
            self.fps = 1 / self.t_kinetic
            self.bin_h = 1
            self.bin_v = 1
            self.cp_h = 2048
            self.cp_w = 2048
            self.start_h = 0
            self.end_h = 2047
            self.start_v = 0
            self.end_v = 2047
            self.pixels_x = 2048
            self.pixels_y = 2048
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 6.5  # micron (read from the datasheet of the sCMOS sensor; not queryable through the SDK)
            self.buffer_size = LIVE_BUFFER_SIZE
            self.acq_num = 0
            self.acq_first = 0
            self.acq_last = 0
            self.valid_index = 0

    def __init__(self, logg=None, camera_index=0):
        self.logg = logg or logger.setup_logging()
        self._settings = self.CameraSettings()
        self.camera_index = camera_index
        self.sdk = None
        self.sensor_width = self.pixels_x
        self.sensor_height = self.pixels_y
        self.data = None
        self.acq_thread = None
        self._frames_read = 0
        self._grabbing = False
        self._serial = None
        self._last_grab_error = None
        self._last_grab_error_time = 0.0
        self._load_sdk_modules()
        self.sdk = self._initialize_sdk()
        if self.sdk is None:
            raise RuntimeError("Revealer sCMOS is not initiated")
        self._configure_camera()

    def __getattr__(self, item):
        if item != "_settings" and hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    # ------------------------------------------------------------------ SDK plumbing
    def _load_sdk_modules(self):
        for path in SCSDK_PATHS:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)
        try:
            self._scsdk = importlib.import_module("scsdk")
            self._scdefs = importlib.import_module("SCDefines")
        except SystemExit:  # scsdk.py calls sys.exit() when the SDK env var is missing; do not let that kill the GUI
            raise RuntimeError(f"Revealer SDK not found: environment variable {SDK_HOME_ENV} is not set")

    def _sdk_attr(self, name):
        """Look a symbol up in scsdk first, then SCDefines (the sample does `from ... import *` on both)."""
        for mod in (self._scsdk, self._scdefs):
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(f"SCSDK python wrapper has no symbol '{name}'")

    def _ok(self, ret, what=""):
        if ret == self._sdk_attr("SC_OK"):
            return True
        self.logg.error(f"Revealer {what} failed, error code {ret}")
        return False

    def _initialize_sdk(self):
        try:
            sdk = self._sdk_attr("SCSDK")()
            ret = sdk.SC_Init(self._sdk_attr("SCLogLevel").Info.value, 'pythonLog', 10485760, 10)
            if not self._ok(ret, "SC_Init"):
                return None
            device_list = self._sdk_attr("SC_DeviceList")()
            device_list.devNum = 0
            device_list.pDevInfo = None
            interface = self._sdk_attr("SC_EInterfaceType").eInterfaceTypeUsb3.value
            ret = self._sdk_attr("SCSDK").SC_EnumDevices(device_list, interface, None)
            if not self._ok(ret, "SC_EnumDevices"):
                return None
            if device_list.devNum == 0:
                self.logg.error("Revealer sCMOS: no device found")
                return None
            if self.camera_index >= device_list.devNum:
                self.logg.error(f"Revealer sCMOS: camera index {self.camera_index} out of range "
                                f"({device_list.devNum} found)")
                return None
            self._serial = device_list.pDevInfo[self.camera_index].serialNumber.decode('utf-8')
            mode = self._sdk_attr("SC_ECreateHandleMode").eModeByIndex
            ret = sdk.SC_CreateHandle(mode, byref(c_void_p(self.camera_index)))
            if not self._ok(ret, "SC_CreateHandle"):
                return None
            ret = sdk.SC_Open()
            if not self._ok(ret, "SC_Open"):
                sdk.SC_DestroyHandle()
                return None
            return sdk
        except Exception as e:
            self.logg.error(f"Error initializing SDK: {e}")
            return None

    def _configure_camera(self):
        self.get_sn()
        self.get_sensor_size()
        self.cooler_on()
        self.set_trigger_mode(1)

    def close(self):
        try:
            if self.acq_thread is not None:
                self.stop_live()
            elif self._grabbing:
                self.stop_snap()
            self.cooler_off()
        finally:
            self._ok(self.sdk.SC_Close(), "SC_Close")
            if self.sdk.handle:
                self._ok(self.sdk.SC_DestroyHandle(), "SC_DestroyHandle")
            try:
                self.sdk.SC_Release()
            except Exception as e:
                self.logg.error(f"SC_Release: {e}")
            self.logg.info("Revealer sCMOS Shut Down")

    # ------------------------------------------------------------------ feature access (GenICam names, see manual)
    # The scsdk.py wrapper takes the feature name as str (it encodes it) and the ctypes out-object itself (it does the
    # byref); do not pass bytes or byref() here.
    def _set_int(self, name, value):
        return self._ok(self.sdk.SC_SetIntFeatureValue(name, int(value)), f"set {name}")

    def _get_int(self, name):
        v = c_int64()
        return v.value if self._ok(self.sdk.SC_GetIntFeatureValue(name, v), f"get {name}") else None

    def _int_range(self, name):
        lo, hi, inc = c_int64(), c_int64(), c_int64()
        if (self._ok(self.sdk.SC_GetIntFeatureMin(name, lo), f"get {name} min") and
                self._ok(self.sdk.SC_GetIntFeatureMax(name, hi), f"get {name} max") and
                self._ok(self.sdk.SC_GetIntFeatureInc(name, inc), f"get {name} inc")):
            return lo.value, hi.value, max(inc.value, 1)
        return None

    def _set_float(self, name, value):
        return self._ok(self.sdk.SC_SetFloatFeatureValue(name, float(value)), f"set {name}")

    def _get_float(self, name):
        v = c_double()
        return v.value if self._ok(self.sdk.SC_GetFloatFeatureValue(name, v), f"get {name}") else None

    def _float_range(self, name):
        lo, hi = c_double(), c_double()
        if (self._ok(self.sdk.SC_GetFloatFeatureMin(name, lo), f"get {name} min") and
                self._ok(self.sdk.SC_GetFloatFeatureMax(name, hi), f"get {name} max")):
            return lo.value, hi.value
        return None

    def _set_bool(self, name, value):
        return self._ok(self.sdk.SC_SetBoolFeatureValue(name, bool(value)), f"set {name}")

    def _set_enum(self, name, value):
        return self._ok(self.sdk.SC_SetEnumFeatureValue(name, int(value)), f"set {name}")

    def _get_enum(self, name):
        v = c_uint64()
        return v.value if self._ok(self.sdk.SC_GetEnumFeatureValue(name, v), f"get {name}") else None

    def software_trigger(self):
        """Fire one frame when TriggerInType is Software_Trigger (5)."""
        return self._ok(self.sdk.SC_ExecuteCommandFeature("TriggerSoftware"), "TriggerSoftware")

    @staticmethod
    def _align(value, lo, hi, inc):
        """Snap an integer feature value down to the camera's increment and clamp it to [lo, hi]."""
        value = lo + ((int(value) - lo) // inc) * inc
        return int(min(max(value, lo), hi))

    # ------------------------------------------------------------------ device info / cooling
    def get_sn(self):
        self.logg.info(f"Camera Serial Number : {self._serial}")

    def get_sensor_size(self):
        w, h = self._get_int("SensorWidth"), self._get_int("SensorHeight")
        if w and h:
            self.sensor_width, self.sensor_height = w, h
            self.pixels_x, self.pixels_y = w, h
            self.cp_w, self.cp_h = w, h
            self.start_h, self.end_h = 0, w - 1
            self.start_v, self.end_v = 0, h - 1
            self.img_size = w * h
            self.logg.info("Detector size: pixels_x = {} pixels_y = {}".format(w, h))

    def cooler_on(self):
        self._set_bool("FanSwitch", True)
        rng = self._int_range("DeviceTemperatureTarget")
        target = COOLING_TARGET if rng is None else self._align(COOLING_TARGET, *rng)
        if self._set_int("DeviceTemperatureTarget", target):
            self.logg.info(f"Revealer Cooler ON, target {target} C")

    def cooler_off(self):
        """Raise the target to the top of its range (the sensor warms up) and keep the fan running."""
        rng = self._int_range("DeviceTemperatureTarget")
        if rng is not None and self._set_int("DeviceTemperatureTarget", rng[1]):
            self.logg.info(f"Revealer Cooler OFF, target {rng[1]} C")

    def get_temperature(self):
        self.temperature = self._get_float("DeviceTemperature")
        self.logg.info("Revealer Temperature {} C".format(self.temperature))

    def check_camera_status(self):
        self.logg.info("Revealer grabbing: {}".format(bool(self.sdk.SC_IsGrabbing())))

    # ------------------------------------------------------------------ settings
    def set_readout_mode(self, ind):
        """
        1 - bit12_STD_High (high speed, high gain)
        2 - bit16_HDR (high dynamic range)
        """
        if self._set_enum("ReadoutMode", ind):
            self.logg.info("Set Readout Mode to {}".format(Readout_Mode.get(ind, ind)))

    def set_shutter_mode(self, ind):
        """0 - Rolling, 1 - GlobalReset, 2 - ProgramableMode"""
        if self._set_enum("ShutterMode", ind):
            self.logg.info("Set Shutter Mode to {}".format(Shutter_Mode.get(ind, ind)))

    def set_trigger_mode(self, ind, activation=0):
        """
        0 - Off
        1 - External_Edge_Trigger
        2 - External_Start_Trigger
        3 - External_Level_Trigger
        4 - Synchronous_Readout
        5 - Software_Trigger
        activation: 0 RisingEdge, 1 FallingEdge, 2 LevelHigh, 3 LevelLow (ignored for Off / Software)
        """
        if not self._set_enum("TriggerInType", ind):
            return
        if ind not in (0, 5):
            self._set_enum("TriggerActivation", activation)
        self.logg.info("Trigger Mode Set to {}".format(Trigger_Mode.get(ind, ind)))

    def set_gain(self):
        """No EM gain on an sCMOS; kept so the executor can call it unconditionally."""
        self.logg.info("sCMOS has no EM gain, gain setting ignored")

    def get_gain(self):
        return self.gain

    def set_roi(self):
        """Apply binning + ROI from bin_h, start_h/end_h, start_v/end_v; pixels_x/y are read back from the camera."""
        if self.bin_h in Binning_Mode:
            self._set_enum("BinningMode", Binning_Mode[self.bin_h])
        else:
            self.logg.error(f"Unsupported binning {self.bin_h}, use 1, 2 or 4")
        want_w = self.end_h - self.start_h + 1
        want_h = self.end_v - self.start_v + 1
        rw, rh = self._int_range("Width"), self._int_range("Height")
        rx, ry = self._int_range("OffsetX"), self._int_range("OffsetY")
        width = self._align(want_w, *rw) if rw else want_w
        height = self._align(want_h, *rh) if rh else want_h
        off_x = self._align(self.start_h, *rx) if rx else self.start_h
        off_y = self._align(self.start_v, *ry) if ry else self.start_v
        # keep the window inside the sensor
        if off_x + width > self.sensor_width:
            off_x = self._align(max(self.sensor_width - width, 0), *rx) if rx else max(self.sensor_width - width, 0)
        if off_y + height > self.sensor_height:
            off_y = self._align(max(self.sensor_height - height, 0), *ry) if ry else max(self.sensor_height - height, 0)
        ret = self.sdk.SC_SetROI(width, height, off_x, off_y)
        if self._ok(ret, "SC_SetROI"):
            self.pixels_x = self._get_int("Width") or width
            self.pixels_y = self._get_int("Height") or height
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 6.5 / self.bin_h
            self.logg.info("bin_h = {} \nbin_v = {} \nstart_h = {} \nstart_v = {} \npixels_x = {} \npixels_y = {}".format(
                self.bin_h, self.bin_v, off_x, off_y, self.pixels_x, self.pixels_y))

    def set_crop(self):
        """Crop = set_roi() on this camera (the sensor window is programmable)."""
        self.set_roi()

    def set_exposure_time(self):
        value = self.t_exposure * EXPOSURE_TIME_SCALE
        rng = self._float_range("ExposureTime")
        if rng is not None:
            value = min(max(value, rng[0]), rng[1])
        if self._set_float("ExposureTime", value):
            actual = self._get_float("ExposureTime")
            self.logg.info("Set Exposure Time to {} (camera reports {})".format(self.t_exposure, actual))

    def set_acquisition_mode(self, ind):
        """The sCMOS free-runs on its trigger input: no acquisition modes to select, kept for API compatibility."""
        self.logg.info("Acquisition mode {} ignored (trigger driven sCMOS)".format(ind))

    def set_kinetic_cycle_time(self, t):
        """Cycle time is set by the trigger sequence; kept for API compatibility."""
        self.t_kinetic = t

    def set_kinetics_num(self, kn):
        """Number of frames is set by the trigger sequence; kept for API compatibility."""
        self.acq_num = kn

    def get_acquisition_timings(self):
        exposure = self._get_float("ExposureTime")
        if exposure is not None:
            self.t_exposure = exposure / EXPOSURE_TIME_SCALE
        fps = self._get_float("AcquisitionFrameRate")
        if fps is not None and fps > 0:
            self.t_kinetic = 1 / fps
            self.fps = 1 / max(self.t_kinetic, self.t_exposure)
            self.t_readout = self.t_kinetic  # whole frame period is treated as dead time after the exposure
        self.logg.info("Get Acquisition Timings exposure = {} kinetic = {} readout = {}".format(
            self.t_exposure, self.t_kinetic, self.t_readout))

    def get_buffer_size(self):
        self.logg.info("Frame buffer = {}".format(self.buffer_size))

    # ------------------------------------------------------------------ frame grabbing
    def _grab_frame(self, timeout_ms):
        """One frame as a copied uint16 (pixels_y, pixels_x) array, or None on timeout / bad frame."""
        frame = self._sdk_attr("SC_Frame")()
        ret = self.sdk.SC_GetFrame(frame, timeout_ms)
        if ret != self._sdk_attr("SC_OK"):
            if ret != self._sdk_attr("SC_TIMEOUT"):  # a timeout is the normal "no new frame yet" answer
                self._log_grab_error(f"SC_GetFrame error code {ret}")
            return None
        try:
            return self._frame_to_array(frame.pData, frame.frameInfo)
        finally:
            self._ok(self.sdk.SC_ReleaseFrame(frame), "SC_ReleaseFrame")

    def _log_grab_error(self, msg):
        """The grab loop runs at thread speed, so repeat the same error at most every 5 s."""
        now = time.monotonic()
        if msg != self._last_grab_error or now - self._last_grab_error_time > 5.0:
            self.logg.error(f"Revealer {msg}")
            self._last_grab_error, self._last_grab_error_time = msg, now

    def _frame_to_array(self, p_data, info):
        w, h = int(info.width), int(info.height)
        bits = (int(info.pixelFormat) >> 16) & 0xFF  # SC_PIX_OCCUPY*BIT: bits stored per pixel
        stride_w, rows = w + int(info.paddingX), h + int(info.paddingY)
        if bits == 16:
            n = int(info.size) // 2
            raw = np.ctypeslib.as_array(cast(p_data, POINTER(c_uint16 * n)).contents)
        elif bits == 8:
            n = int(info.size)
            raw = np.ctypeslib.as_array(cast(p_data, POINTER(c_ubyte * n)).contents).astype(np.uint16)
        else:
            self._log_grab_error(f"pixel format 0x{int(info.pixelFormat):08X} stores {bits} bit per pixel; "
                                 f"set a 16-bit PixelFormat / ReadoutMode (packed formats are not supported)")
            return None
        if raw.size < stride_w * rows:
            self._log_grab_error(f"frame has {raw.size} pixels, expected {stride_w * rows} ({w}x{h} + padding)")
            return None
        return raw[:stride_w * rows].reshape(rows, stride_w)[:h, :w].copy()

    def _set_buffer_count(self, n):
        n = int(min(max(n, MIN_BUFFER_COUNT), MAX_BUFFER_COUNT))
        if self._ok(self.sdk.SC_SetBufferCount(n), "SC_SetBufferCount"):
            self.buffer_size = n

    def _start_grabbing(self):
        self._frames_read = 0
        if not self._ok(self.sdk.SC_StartGrabbing(), "SC_StartGrabbing"):
            raise RuntimeError("Revealer SC_StartGrabbing failed")
        self._grabbing = True

    def _stop_grabbing(self):
        if self._grabbing:
            self._ok(self.sdk.SC_StopGrabbing(), "SC_StopGrabbing")
            self._grabbing = False

    def get_images(self):
        """Called in a loop by CameraAcquisitionThread: move whatever frames are ready into self.data."""
        if self.data is None:
            return
        frames = []
        frame = self._grab_frame(GRAB_TIMEOUT_MS)
        while frame is not None:
            frames.append(frame)
            if len(frames) >= MAX_FRAMES_PER_POLL:
                break
            frame = self._grab_frame(DRAIN_TIMEOUT_MS)
        if not frames or self.data is None:
            return
        first = self._frames_read
        self._frames_read += len(frames)
        self.data.add_element(frames, [first, self._frames_read - 1])

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element()
        else:
            return None

    def prepare_live(self, aq=None):
        self.set_roi()
        self.set_exposure_time()
        self.buffer_size = LIVE_BUFFER_SIZE
        self.get_acquisition_timings()
        self.get_buffer_size()

    def start_live(self):
        self.data = run_threads.CameraDataList(max_length=self.buffer_size)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)
        self._set_buffer_count(MAX_BUFFER_COUNT)
        self._start_grabbing()
        self.acq_thread.start()
        self.logg.info('Start live image')

    def stop_live(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()
            self.data = None
        self._stop_grabbing()
        self.logg.info('Live image stopped')

    def start_snap(self):
        self._set_buffer_count(MIN_BUFFER_COUNT)
        self._start_grabbing()
        self.logg.info('Start snap shot')

    def stop_snap(self):
        self._stop_grabbing()
        self.logg.info('Snap shot stopped')

    def get_last_snap(self, timeout=1.2):
        """Wait for a frame (the trigger fires it) and return the newest one that is waiting."""
        deadline = time.monotonic() + timeout
        last = None
        while last is None:
            remaining_ms = int(max(deadline - time.monotonic(), 0) * 1000)
            last = self._grab_frame(max(min(remaining_ms, 100), 1))
            if last is None and time.monotonic() >= deadline:
                self.logg.error("Timeout waiting for new camera frame")
                return None
        for _ in range(MAX_FRAMES_PER_POLL):  # bounded: a free-running camera would never run dry
            newer = self._grab_frame(DRAIN_TIMEOUT_MS)
            if newer is None:
                break
            last = newer
        return last

    def check_new_acquisition(self):
        return True if self._grabbing else None

    def prepare_data_acquisition(self, aq=None, preset=None):
        self.set_roi()
        self.set_exposure_time()
        self.get_acquisition_timings()
        self.get_buffer_size()

    def start_data_acquisition(self, n, fd, fn):
        self.data = run_threads.CameraDataList(max_length=n, save_to_disk=True, save_dir=fd, file_prefix=fn)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)
        self._set_buffer_count(2 * n)
        self._start_grabbing()
        self.acq_thread.start()
        self.logg.info('Acquisition started')

    def stop_data_acquisition(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()  # flush + join the background TIFF-writer thread
            self.data = None
        self._stop_grabbing()
        self.logg.info('Acquisition stopped')

    def get_data(self):
        if self.data is not None:
            return self.data.get_elements()
        else:
            return None

    def get_acq_num(self):
        self.acq_first, self.acq_last = 0, max(self._frames_read - 1, 0)
        self.logg.info(f"{self.acq_first} {self.acq_last}")

    def check_acquisition_progress(self):
        self.logg.info("frames received = {}".format(self._frames_read))

    def wait_for_acquisition(self):
        while self._grabbing and self.data is not None and self._frames_read < self.data.max_length:
            time.sleep(0.01)

    def free_memory(self):
        """SDK owns the frame buffers (released per frame in _grab_frame); nothing to free."""
        pass
