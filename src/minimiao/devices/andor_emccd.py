# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import sys
import numpy as np
from pyAndorSDK2 import atmcd, atmcd_errors
from minimiao import run_threads, logger

sys.path.append(r'C:\Program Files\Andor SDK')

Preset_Mode = {
    0: "Sensitivity and Speed (EM Amplifier)",
    1: "Dynamic Range and Speed (EM Amplifier)",
    2: "Fastest Frame Rate (EM Amplifier)",
    3: "Time Lapse (EM Amplifier)",
    4: "Time Lapse and Short Exposures (EM Amplifier)",
    5: "EMCCD Highest Dynamic Range (EM Amplifier)",
    6: "CCD Lowest Noise / Slow readout"
}

Readout_Mode = {0: "Full Vertical Binning",
                1: "Multi-Track",
                2: "Random-Track",
                3: "Single-Track",
                4: "Image"}

Trigger_Mode = {0: "Internal",
                1: "External",
                6: "External Start",
                7: "External Exposure",
                10: "Software"}

Acquisition_Mode = {1: "Single Scan",
                    2: "Accumulate",
                    3: "Kinetics",
                    4: "Fast Kinetics",
                    5: "Run Till Abort"}


class EMCCDCamera:
    class CameraSettings:
        def __init__(self):
            self.temperature = None
            self.gain = 0
            self.t_clean = 0.0
            self.t_readout = 0.04
            self.t_exposure = 0.01
            self.t_accumulate = None
            self.t_kinetic = 0.05
            self.fps = 1 / self.t_kinetic
            self.bin_h = 1
            self.bin_v = 1
            self.cp_h = 1024
            self.cp_w = 1024
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
        self.logg = logg or logger.setup_logging()
        self._settings = self.CameraSettings()
        self.sdk = self._initialize_sdk()
        if self.sdk:
            self._configure_camera()
        self.preset_modes = None
        self.data = None
        self.acq_thread = None

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _initialize_sdk(self):
        try:
            sdk = atmcd(r'C:\Program Files\Andor SDK')
            ret = sdk.Initialize(r'C:/Program Files/Andor SDK/atmcd64d.dll')
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                return sdk
            else:
                self.logg.error('Andor EMCCD is not initiated')
                return None
        except Exception as e:
            self.logg.error(f"Error initializing SDK: {e}")
            return None

    def _configure_camera(self):
        self.get_sn()
        self.cooler_on()

    def close(self):
        self.cooler_off()
        ret = self.sdk.ShutDown()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Andor EMCCD Shut Down")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def load_preset_modes(self):
        preset_fn = r"C:\ProgramData\AndorSolis\usermodes.xml"
        preset_fnb = preset_fn.encode("mbcs")
        ret = self.sdk.OA_Initialize(preset_fnb, len(preset_fnb))
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(f"Camera Preset Acquisition Mode Loaded")
            ret, mds = self.sdk.OA_GetUserModeNames()
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                raw_bytes = mds.value
                text = raw_bytes.decode("utf-8", errors="replace")
                mode_list = [x.strip() for x in text.split(',') if x.strip()]
                self.logg.info(f"Camera Preset Acquisition Mode Available: {mode_list}")
                self.preset_modes = mode_list
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
                self.preset_modes = []
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
            self.preset_modes = []

    def set_preset_mode(self, ind):
        mode_name = self.preset_modes[ind]
        mode_name_b = mode_name.encode("mbcs")  # Windows encoding for Andor SDK2 char*
        ret = self.sdk.OA_EnableMode(mode_name_b)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(f"Camera Acquisition Mode set to Preset : {mode_name}")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_sn(self):
        ret, serial_number = self.sdk.GetCameraSerialNumber()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(f"Camera Serial Number : {serial_number}")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def cooler_on(self):
        ret = self.sdk.SetTemperature(-60)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            ret = self.sdk.CoolerON()
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.logg.info("EMCCD Cooler ON")
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def cooler_off(self):
        ret = self.sdk.CoolerOFF()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("EMCCD Cooler OFF")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_temperature(self):
        ret, self.temperature = self.sdk.GetTemperature()
        self.logg.info("EMCCD Temperature {} C".format(self.temperature))

    def check_camera_status(self):
        ret, status = self.sdk.GetStatus()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(atmcd_errors.Error_Codes(status))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_sensor_size(self):
        ret, self.pixels_x, self.pixels_y = self.sdk.GetDetector()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.img_size = self.pixels_x * self.pixels_y
            self.logg.info("Detector size: pixels_x = {} pixels_y = {}".format(self.pixels_x, self.pixels_y))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_readout_mode(self, ind):
        """
        0 - Full Vertical Binning
        1 - Multi-Track
        2 - Random-Track
        3 - Single-Track
        4 - Image
        """
        ret = self.sdk.SetReadMode(ind)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Readout Mode to {}".format(Readout_Mode[ind]))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_single_track(self, center=512, width=64):
        """Need shutter to prevent light falling outside the selected lines"""
        ret = self.sdk.SetReadMode(3)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            ret = self.sdk.SetSingleTrack(centre=center, height=width)
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.logg.info("Set Readout Mode to {}".format(Readout_Mode[3]))
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_readout_rate(self, va=0, hs=0, vs=0):
        ret = self.sdk.SetVSAmplitude(va)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Vertical Clock Voltage {}  ".format(va))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret = self.sdk.SetHSSpeed(0, hs)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            (ret, speed) = self.sdk.GetHSSpeed(0, 0, hs)
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.logg.info("HSSpeeds {} MHz  ".format(speed))
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret = self.sdk.SetVSSpeed(vs)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            (ret, speed) = self.sdk.GetVSSpeed(vs)
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.logg.info("VSSpeeds {} us/pixel  ".format(speed))
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_frame_transfer(self, ft=1):
        ret = self.sdk.SetFrameTransferMode(ft)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Frame Transfer Mode {}".format(ft))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_gain(self):
        ret = self.sdk.SetEMCCDGain(self.gain)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.get_gain()
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_gain(self):
        ret, self.gain = self.sdk.GetEMCCDGain()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("CCD EMGain is {}".format(self.gain))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_roi(self):
        ret = self.sdk.SetImage(self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("bin_h = {} \nbin_v = {} \nstart_h = {} \nend_h = {} \nstart_v = {} \nend_v = {}".format(
                self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v))
            self.pixels_x = self.end_h - self.start_h + 1
            self.pixels_y = self.end_v - self.start_v + 1
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 13 / self.bin_h
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_crop(self):
        ret = self.sdk.SetIsolatedCropMode(1, self.cp_h, self.cp_w, self.bin_h, self.bin_v)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("bin_h = {} \nbin_v = {} \nstart_h = {} \nend_h = {} \nstart_v = {} \nend_v = {}".format(
                self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v))
            self.pixels_x = self.end_h - self.start_h + 1
            self.pixels_y = self.end_v - self.start_v + 1
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 13 / self.bin_h
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_trigger_mode(self, ind):
        """
        0 - Internal
        1 - External
        6 - External Start
        7 - External Exposure
        10 - Software
        """
        ret = self.sdk.SetTriggerMode(ind)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Trigger Mode Set to {}".format(Trigger_Mode[ind]))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_exposure_time(self):
        ret = self.sdk.SetExposureTime(self.t_exposure)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Exposure Time to {}".format(self.t_exposure))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_acquisition_mode(self, ind):
        """
        1 - Single Scan
        2 - Accumulate
        3 - Kinetics
        4 - Fast Kinetics
        5 - Run Till Abort
        """
        ret = self.sdk.SetAcquisitionMode(ind)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Acquisition Mode to {}".format(Acquisition_Mode[ind]))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_acquisition_timings(self):
        ret, self.t_exposure, self.t_accumulate, self.t_kinetic = self.sdk.GetAcquisitionTimings()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Acquisition Timings exposure = {} accumulate = {} kinetic = {}".format(self.t_exposure,
                                                                                                       self.t_accumulate,
                                                                                                       self.t_kinetic))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret, self.t_readout = self.sdk.GetReadOutTime()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Readout Time = {}".format(self.t_readout))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret, self.t_clean = self.sdk.GetKeepCleanTime()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Keep Clean Time = {}".format(self.t_clean))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_buffer_size(self):
        ret, self.buffer_size = self.sdk.GetSizeOfCircularBuffer()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Circular Buffer = {}".format(self.buffer_size))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_kinetic_cycle_time(self, t):
        ret = self.sdk.SetKineticCycleTime(t)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Kinetic Cycle Time to {}".format(t))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_kinetics_num(self, kn):
        ret = self.sdk.SetNumberKinetics(kn)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Number of Kinetics to {}".format(kn))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def prepare_live(self, rd=4, aq=5, tr=1, preset=2):
        self.set_readout_mode(rd)
        self.set_roi()
        self.set_trigger_mode(tr)
        self.set_acquisition_mode(aq)
        self.set_frame_transfer(1)
        self.set_preset_mode(preset)
        self.set_exposure_time()
        self.set_gain()
        self.get_acquisition_timings()
        self.get_buffer_size()

    def start_live(self):
        self.data = run_threads.CameraDataList(max_length=self.buffer_size)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)
        ret = self.sdk.PrepareAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            ret = self.sdk.StartAcquisition()
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.acq_thread.start()
                self.logg.info('Start live image')
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def stop_live(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()
            self.data = None
        ret = self.sdk.AbortAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Live image stopped')
            self.free_memory()
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_images(self):
        if self.data is None:
            return

        ret, first, last = self.sdk.GetNumberNewImages()
        if ret != atmcd_errors.Error_Codes.DRV_SUCCESS:
            return

        num = last - first + 1
        if num <= 0:
            return

        ret, data_array, valid_first, valid_last = self.sdk.GetImages16(first, last, self.img_size * num)
        if ret != atmcd_errors.Error_Codes.DRV_SUCCESS:
            return

        if self.data is None:
            return

        frames = np.asarray(data_array, dtype=np.uint16).reshape(num, self.pixels_y, self.pixels_x)
        self.data.add_element([frames[i] for i in range(num)], [valid_first, valid_last])

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element()
        else:
            return None

    def prepare_data_acquisition(self, rd=4, aq=5, tr=1, preset=2):
        self.set_readout_mode(rd)
        self.set_roi()
        self.set_trigger_mode(tr)
        self.set_acquisition_mode(aq)
        self.set_frame_transfer(1)
        self.set_preset_mode(preset)
        self.set_exposure_time()
        self.set_gain()
        self.get_acquisition_timings()
        self.get_buffer_size()

    def start_data_acquisition(self, n, fd, fn):
        self.data = run_threads.CameraDataList(max_length=n, save_to_disk=True, save_dir=fd, file_prefix=fn)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)
        ret = self.sdk.PrepareAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            ret = self.sdk.StartAcquisition()
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.acq_thread.start()
                self.logg.info('Acquisition started')
            else:
                self.logg.error(atmcd_errors.Error_Codes(ret))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def stop_data_acquisition(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()   # flush + join the background TIFF-writer thread
            self.data = None
        ret = self.sdk.AbortAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Acquisition stopped')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        self.free_memory()

    def get_data(self):
        if self.data is not None:
            return self.data.get_elements()
        else:
            return None

    def save_as_sif(self, filename):
        ret = self.sdk.SaveAsSif(filename)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Data Saved as Sif')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_acq_num(self):
        ret, first, last = self.sdk.GetNumberAvailableImages()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.acq_first, self.acq_last = first, last
            self.logg.info(first, last)
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def check_acquisition_progress(self):
        ret, self.numAccumulate, self.numKinetics = self.sdk.GetAcquisitionProgress()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(
                "GetAcquisitionProgress returned {} \n"
                "number of accumulations completed = {} \n"
                "kinetic scans completed = {}".format(ret, self.numAccumulate, self.numKinetics))

    def wait_for_acquisition(self):
        ret = self.sdk.WaitForAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Waiting for Acquisition')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def free_memory(self):
        ret = self.sdk.FreeInternalMemory()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Internal Memory Free')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
