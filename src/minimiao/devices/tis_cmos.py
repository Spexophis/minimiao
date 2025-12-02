import sys

sys.path.append(r'C:\Program Files\The Imaging Source Europe GmbH\sources')

import ctypes
import numpy as np
from collections import deque
import tisgrabber as tis

ic = ctypes.cdll.LoadLibrary(r'C:\Program Files\The Imaging Source Europe GmbH\sources\tisgrabber_x64.dll')
tis.declareFunctions(ic)
ic.IC_InitLibrary(0)


class CallbackData(ctypes.Structure):
    def __init__(self, max_length):
        super().__init__()
        self.image_data = deque(maxlen=max_length)
        self.image_counter = 0

    def add_element(self, element):
        self.image_data.extend(element)

    def get_elements(self):
        return np.array(self.image_data) if self.image_data else None

    def get_last_element(self):
        return self.image_data[-1].copy() if self.image_data else None

    def is_empty(self):
        return len(self.image_data) == 0


def get_buffer(hGrabber):
    img_width = ctypes.c_long()
    img_height = ctypes.c_long()
    img_depth = ctypes.c_int()
    color_format = ctypes.c_int()
    if ic.IC_GetImageDescription(hGrabber, img_width, img_height, img_depth, color_format) == tis.IC_SUCCESS:
        img_depth = int(img_depth.value / 16.0) * ctypes.sizeof(ctypes.c_uint16)
        buffer_size = img_width.value * img_height.value * img_depth
        return buffer_size, img_width, img_height, img_depth
    else:
        return None, None, None, None


def get_data(image_pointer, buffer_size, width, height):
    image_data = ctypes.cast(image_pointer, ctypes.POINTER(ctypes.c_ubyte * int(buffer_size)))
    image = np.ndarray(shape=(height.value, width.value), buffer=image_data.contents, dtype=np.uint16)
    return image


def frame_ready_callback(hGrabber, image_pointer, frame_number, qdata):
    """ Callback function run after every trigger to collect the image and save it in the pData class."""
    buffer_size, img_width, img_height, img_depth = get_buffer(hGrabber)
    if buffer_size is not None and buffer_size > 0:
        image = get_data(image_pointer, buffer_size, img_width, img_height)
        qdata.add_element([image])
        qdata.image_counter += 1


class TISCamera:
    class CameraSettings:
        def __init__(self):
            self.t_clean = None
            self.t_readout = None
            self.t_exposure = None
            self.t_accumulate = None
            self.t_kinetic = None
            self.bin_h = 1
            self.bin_v = 1
            self.start_h = 1
            self.end_h = 1024
            self.start_v = 1
            self.end_v = 1024
            self.pixels_x = 1024
            self.pixels_y = 1024
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 13  # micron
            self.buffer_size = None
            self.acq_num = 0
            self.acq_first = 0
            self.acq_last = 0
            self.valid_index = 0

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self._settings = self.CameraSettings()
        self.data = CallbackData(8)
        self.frame_ready = ic.FRAMEREADYCALLBACK(frame_ready_callback)
        self.hGrabber = self._initialize_grabber()
        if self.hGrabber is not None:
            self._configure_camera()

    def __del__(self):
        pass

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def close(self):
        r = ic.IC_CloseVideoCaptureDevice(self.hGrabber)
        if r:
            r = ic.IC_ReleaseGrabber(self.hGrabber)
            if r:
                ic.IC_CloseLibrary()
                self.logg.info("TIS Camera OFF")

    def _initialize_grabber(self):
        device_count = ic.IC_GetDeviceCount()
        if device_count == 1:
            unique_name = tis.D(ic.IC_GetUniqueNamefromList(0))
            self.logg.info("Device {}".format(tis.D(ic.IC_GetDevice(0))))
            self.logg.info("Unique Name : {}".format(unique_name))
            grabber = ic.IC_CreateGrabber()
            if ic.IC_OpenDevByUniqueName(grabber, tis.T(unique_name)) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: TIS Camera ON")
                if ic.IC_IsDevValid(grabber):
                    self.filters = self._create_frame_filters(grabber)
                    self.logg.info("hGrabber initialized")
                    return grabber
                else:
                    self.logg.error('hGrabber is invalid')
                    return None
            else:
                self.logg.error(f"Error opening TIS camera")
                return None
        else:
            self.logg.error('No device found')
            return None

    def _configure_camera(self):
        if ic.IC_SetFormat(self.hGrabber, tis.SinkFormats.Y16.value) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Format Y16")
        else:
            self.logg.error("FAIL: Set Format Y16")
        if ic.IC_SetVideoFormat(self.hGrabber, tis.T("Y16 (2448x2048)")) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Video Format Y16 (2448x2048)")
        else:
            self.logg.error("FAIL: Set Video Format Y16 (2448x2048)")
        if ic.IC_SetFrameRate(self.hGrabber, ctypes.c_float(37.5)) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Frame Rate 37.5fps")
        else:
            self.logg.error("FAIL: Set Frame Rate 37.5fps")
        if ic.IC_SetContinuousMode(self.hGrabber, 0) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Continuous Mode ON")
        else:
            self.logg.error("FAIL: Set Continuous Mode ON")
        if ic.IC_SetPropertyValue(self.hGrabber, tis.T("Brightness"), tis.T("Value"),
                                  ctypes.c_int(0)) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Brightness zero")
        else:
            self.logg.error("FAIL: Set Brightness zero")
        if ic.IC_SetPropertyValue(self.hGrabber, tis.T("Gain"), tis.T("Value"), ctypes.c_int(0)) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Gain zero")
        else:
            self.logg.error("FAIL: Set Gain zero")
        if ic.IC_SetPropertyValue(self.hGrabber, tis.T("Denoise"), tis.T("Value"), ctypes.c_int(16)) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Denoise to 16")
        else:
            self.logg.error("FAIL: Set Denoise")
        if ic.IC_SetFrameReadyCallback(self.hGrabber, self.frame_ready, self.data) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Set Frame Ready Callback")
        else:
            self.logg.error("FAIL: Set Frame Ready Callback")
        self.set_trigger_mode(sw=False)
        self.set_denoise(2)

    def _create_frame_filters(self, grabber):
        filter_handles = {"DeNoise": None, "ROI": None}
        filter_handle = tis.HFRAMEFILTER()
        if ic.IC_CreateFrameFilter(tis.T("DeNoise"), filter_handle) == tis.IC_SUCCESS:
            self.logg.info("DeNoise filter loaded.")
            if ic.IC_AddFrameFilterToDevice(grabber, filter_handle) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Add DeNoise Filter")
                filter_handles["DeNoise"] = filter_handle
            else:
                self.logg.error("FAIL: Add DeNoise Filter")
        else:
            self.logg.error("DeNoise filter load failed")
        filter_handle = tis.HFRAMEFILTER()
        if ic.IC_CreateFrameFilter(tis.T("ROI"), filter_handle) == tis.IC_SUCCESS:
            self.logg.info("ROI filter loaded.")
            if ic.IC_AddFrameFilterToDevice(grabber, filter_handle) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Add ROI Filter")
                filter_handles["ROI"] = filter_handle
            else:
                self.logg.error("FAIL: Add ROI Filter")
        else:
            self.logg.error("ROI filter load failed")
        return filter_handles

    def set_roi(self, left, top, width, height):
        if self.filters["ROI"] is not None:
            if 0 <= left < 2448:
                if ic.IC_FrameFilterSetParameterInt(self.filters["ROI"], tis.T("Left"), left) == tis.IC_SUCCESS:
                    self.logg.info(f"SUCCESS: Set ROI Left to {left}")
                else:
                    self.logg.error(f"FAIL: Set ROI Left to {left}")
            if 0 <= top < 2048:
                if ic.IC_FrameFilterSetParameterInt(self.filters["ROI"], tis.T("Top"), top) == tis.IC_SUCCESS:
                    self.logg.info(f"SUCCESS: Set ROI Top to {top}")
                else:
                    self.logg.error(f"FAIL: Set ROI Top to {top}")
            if left + width <= 2448:
                if ic.IC_FrameFilterSetParameterInt(self.filters["ROI"], tis.T("Width"), width) == tis.IC_SUCCESS:
                    self.logg.info(f"SUCCESS: Set ROI Width to {width}")
                else:
                    self.logg.error(f"FAIL: Set ROI Width to {width}")
            if top + height <= 2048:
                if ic.IC_FrameFilterSetParameterInt(self.filters["ROI"], tis.T("Height"), height) == tis.IC_SUCCESS:
                    self.logg.info(f"SUCCESS: Set ROI Height to {height}")
                else:
                    self.logg.error(f"FAIL: Set ROI Height to {height}")

    def set_denoise(self, level):
        if self.filters["DeNoise"] is not None:
            if ic.IC_FrameFilterSetParameterInt(self.filters["DeNoise"], tis.T("DeNoise Level"),
                                                level) == tis.IC_SUCCESS:
                self.logg.info(f"SUCCESS: Set DeNoise Filter to {level}")
            else:
                self.logg.error(f"FAIL: Set DeNoise Filter to {level}")

    def prepare_live(self):
        if ic.IC_PrepareLive(self.hGrabber, 0):
            self.logg.info('Live ready')

    def start_live(self):
        if ic.IC_IsDevValid(self.hGrabber):
            if ic.IC_StartLive(self.hGrabber, 0) == tis.IC_SUCCESS:
                self.logg.info('Live start')

    def suspend_live(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_SuspendLive(self.hGrabber)

    def stop_live(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_StopLive(self.hGrabber)
            self.logg.info('Live stop')

    def send_trigger(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_SoftwareTrigger(self.hGrabber)
            self.logg.info('Software Trigger')

    def set_trigger_mode(self, sw=True, expo=None):
        if sw:
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("Enable"), 1) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Set Trigger Enable")
            else:
                self.logg.error("FAIL: Set Trigger Enable")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("Polarity"), 0) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Set Trigger Polarity")
            else:
                self.logg.error("FAIL: Set Trigger Polarity")
            if expo is not None:
                if ic.IC_SetPropertyMapString(self.hGrabber, tis.T("Trigger"), tis.T("Exposure Mode"),
                                              tis.T("Timed")) == tis.IC_SUCCESS:
                    self.logg.info("SUCCESS: Set Trigger Exposure Mode Timed")
                else:
                    self.logg.error("FAIL: Set Trigger Exposure Mode Timed")
                self.set_exposure(expo)
            else:
                if ic.IC_SetPropertyMapString(self.hGrabber, tis.T("Trigger"), tis.T("Exposure Mode"),
                                              tis.T("Trigger Width")) == tis.IC_SUCCESS:
                    self.logg.info("SUCCESS: Set Trigger Exposure Mode Trigger Width")
                else:
                    self.logg.error("FAIL: Set Trigger Exposure Mode Trigger Width")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("IMX Low-Latency Mode"),
                                       1) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Set Trigger IMX Low-Latency Mode")
            else:
                self.logg.error("FAIL: Set Trigger IMX Low-Latency Mode")
            if ic.IC_RemoveOverlay(self.hGrabber, 1) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Remove Overlay")
            else:
                self.logg.error("FAIL: Remove Overlay")
        else:
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("Enable"), 0) == tis.IC_SUCCESS:
                self.logg.info("SUCCESS: Set Trigger Disable")
            else:
                self.logg.error("FAIL: Set Trigger Disable")

    def get_gain(self, verbose=True):
        if ic.IC_IsDevValid(self.hGrabber):
            gain_min = ctypes.c_long()
            gain_max = ctypes.c_long()
            gain = ctypes.c_long()
            ic.IC_GetPropertyValue(self.hGrabber, tis.T("Gain"), tis.T("Value"), gain)
            ic.IC_GetPropertyValueRange(self.hGrabber, tis.T("Gain"), tis.T("Value"),
                                        gain_min, gain_max)
            self.logg.info("Gain is {0} range is {1} - {2}".format(gain.value, gain_min.value, gain_max.value))
            if verbose:
                return gain.value

    def set_gain(self, gain):
        if ic.IC_IsDevValid(self.hGrabber):
            r = ic.IC_SetPropertyAbsoluteValue(self.hGrabber, tis.T("Gain"), tis.T("Value"), ctypes.c_float(gain))
            if r:
                self.get_gain(False)

    def get_exposure(self, verbose=True):
        if ic.IC_IsDevValid(self.hGrabber):
            expo_min = ctypes.c_float()
            expo_max = ctypes.c_float()
            exposure = ctypes.c_float()
            ic.IC_GetPropertyAbsoluteValue(self.hGrabber, tis.T("Exposure"), tis.T("Value"), exposure)
            ic.IC_GetPropertyAbsoluteValueRange(self.hGrabber, tis.T("Exposure"), tis.T("Value"), expo_min, expo_max)
            self.logg.info("Exposure is {0}, range is {1} - {2}".format(exposure.value, expo_min.value, expo_max.value))
            if verbose:
                return exposure.value

    def set_exposure(self, exposure):
        if ic.IC_IsDevValid(self.hGrabber):
            r = ic.IC_SetPropertyAbsoluteValue(self.hGrabber, tis.T("Exposure"), tis.T("Value"),
                                               ctypes.c_float(exposure))
            if r:
                self.get_exposure(False)

    def snap_image(self):
        if ic.IC_SnapImage(self.hGrabber, 2000) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Image Capture")
            return True
        else:
            self.logg.error("FAIL: Image Capture")
            return False

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element()
        else:
            return None

    def get_data(self):
        if self.data is not None:
            return self.data.get_elements()
        else:
            return None

    def show_property(self):
        ic.IC_ShowPropertyDialog(self.hGrabber)

    def save_img(self):
        if ic.IC_SaveImage(self.hGrabber, tis.T(r'C:\Users\ruizhe.lin\Desktop\test.jpg'), tis.ImageFileTypes['JPEG'],
                           100) == tis.IC_SUCCESS:
            self.logg.info("SUCCESS: Image Save")
        else:
            self.logg.error("FAIL: Image Save")
