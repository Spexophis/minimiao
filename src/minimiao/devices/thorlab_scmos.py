# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
from minimiao import run_threads, logger

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

path_to_files = r"C:\Program Files\Thorlabs\Scientific Camera Interfaces\SDK\Python Toolkit"


class ThorCMOS:
    class CameraSettings:
        def __init__(self):
            self.t_clean = 0.
            self.t_readout = 0.004
            self.t_exposure = 0
            self.bin_h = 1
            self.bin_v = 1
            self.start_h = 0
            self.end_h = 2447
            self.start_v = 0
            self.end_v = 2047
            self.pixels_x = 2448
            self.pixels_y = 2048
            self.ps = 3.45  # micron
            self.acq_num = 0
            self.acq_first = 0
            self.acq_last = 0
            self.valid_index = 0

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self._settings = self.CameraSettings()
        try:
            self._configure_path()
        except Exception as e:
            self.logg.error(f"Failed to load ThorCMOS DLLs {e}")
            return
        self.sdk, serial = self._init_sdk()
        if self.sdk is not None:
            try:
                self.camera = self._init_cam(serial)
            except Exception as e:
                self.camera = None
                self.logg.error(f"{e}")
                return
        if self.camera is not None:
            self._config_cam()
        self.data = None
        self.acq_thread = None

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @staticmethod
    def _configure_path():
        relative_path_to_dlls = '.' + os.sep + 'dlls' + os.sep
        relative_path_to_dlls += '64_lib'
        absolute_path_to_dlls = os.path.abspath(path_to_files + os.sep + relative_path_to_dlls)
        os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
        try:
            os.add_dll_directory(absolute_path_to_dlls)
        except AttributeError:
            raise

    def _init_sdk(self):
        sdk = TLCameraSDK()
        available_cameras = sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            self.logg.error(f"No ThorCMOS found")
            return None, None
        else:
            self.logg.info(f"Found {len(available_cameras)} ThorCMOS {available_cameras[0]}")
            return sdk, available_cameras[0]

    def _init_cam(self, serial):
        try:
            camera = self.sdk.open_camera(serial)
            self.logg.info(f"ThorCMOS initiated successfully")
            return camera
        except Exception as e:
            raise RuntimeError(f"Opening the ThorCMOS failed with error code {e}")

    def _config_cam(self):
        self.camera.frame_rate_control_value = 50
        self.camera.is_frame_rate_control_enabled = True
        self.camera.frames_per_trigger_zero_for_unlimited = 1
        self.camera.image_poll_timeout_ms = 0
        self.set_acquisition_mode(2)
        self.set_trigger_polarity(0)

    def close(self):
        if self.camera.is_armed:
            self.camera.disarm()
        self.camera.dispose()
        self.sdk.dispose()

    def set_roi(self):
        self.camera.binx = self.bin_h
        self.camera.biny = self.bin_v
        self.camera.roi = (self.start_h, self.start_v, self.end_h, self.end_v)
        self.pixels_x = self.end_h - self.start_h + 1
        self.pixels_y = self.end_v - self.start_v + 1
        self.img_size = self.pixels_x * self.pixels_y
        self.ps = 3.45 / self.bin_h
        self.logg.info("bin_h = {} \nbin_v = {} \nstart_h = {} \nend_h = {} \nstart_v = {} \nend_v = {}".format(
            self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v))

    def set_frame_rate(self, frame_rate):
        self.camera.frame_rate_control_value = frame_rate
        self.camera.is_frame_rate_control_enabled = True

    def set_exposure_time(self):
        self.camera.exposure_time_us = int(self.t_exposure * 1e6)
        self.logg.info("Set Exposure Time to {}".format(self.t_exposure))

    def set_acquisition_mode(self, md):
        """
        0: SOFTWARE TRIGGER;
        1: HARDWARE TRIGGER;
        2: BULB (TRIGGER EXPOSURE)
        """
        self.camera.operation_mode = md

    def set_trigger_polarity(self, tp):
        """
        0: ACTIVE_HIGH (rising edge);
        1: ACTIVE_LOW (falling edge)
        """
        self.camera.trigger_polarity = tp

    def send_software_trigger(self):
        self.camera.issue_software_trigger()

    def prepare_live(self, max_length=64):
        self.set_roi()
        self.set_exposure_time()
        self.data = run_threads.DPCCameraDataList(max_length=max_length)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)

    def start_live(self):
        self.camera.arm(4)
        self.acq_thread.start()

    def stop_live(self):
        self.camera.disarm()
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()
            self.data = None

    def prepare_data_acquisition(self, n, fd, fn):
        self.set_roi()
        self.set_exposure_time()
        self.data = run_threads.DPCCameraDataList(max_length=n, save_to_disk=True, save_dir=fd, file_prefix=fn)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)

    def start_data_acquisition(self, buffer_size=16):
        self.camera.arm(buffer_size)
        self.acq_thread.start()
        self.logg.info("Acquisition started")

    def stop_data_acquisition(self):
        self.camera.disarm()
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()
            self.data = None
        self.logg.info("Acquisition stopped")

    def get_images(self):
        if self.data is None:
            return
        while True:
            frame = self.camera.get_pending_frame_or_null()
            if frame is None:
                break
            image_buffer_copy = np.copy(frame.image_buffer)
            numpy_shaped_image = image_buffer_copy.reshape(self.camera.image_height_pixels,
                                                           self.camera.image_width_pixels)
            self.data.add_element([numpy_shaped_image], [frame.frame_count, frame.frame_count])

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
