# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from pyvcam import pvc
from pyvcam.camera import Camera
from pyvcam import constants as pvc_consts
from minimiao import run_threads, logger


# --- Exposure mode constants (PARAM_EXPOSURE_MODE) ---
# The expose-out mode is OR-ed with the trigger mode:
#   cam.exp_mode = trigger_mode | (expose_out_mode << 8)
# Simple trigger modes (expose-out = FIRST_ROW by default):
Exposure_Mode = {
    pvc_consts.TIMED_MODE:         "Internal (Timed)",
    pvc_consts.STROBED_MODE:       "External Edge Trigger",
    pvc_consts.BULB_MODE:          "External Level / Bulb",
    pvc_consts.TRIGGER_FIRST_MODE: "Trigger First Frame Only",
    pvc_consts.EXT_TRIG_INTERNAL:  "External Trigger, Internal Timing",
    pvc_consts.EXT_TRIG_EDGE_RISING: "External Rising-Edge Trigger",
    pvc_consts.EXT_TRIG_LEVEL:     "External Level Trigger",
}

# --- Expose-out (output signal) modes (PARAM_EXPOSE_OUT_MODE) ---
Expose_Out_Mode = {
    pvc_consts.EXPOSE_OUT_FIRST_ROW:      "First Row",
    pvc_consts.EXPOSE_OUT_ALL_ROWS:       "All Rows",
    pvc_consts.EXPOSE_OUT_ANY_ROW:        "Any Row",
    pvc_consts.EXPOSE_OUT_ROLLING_SHUTTER:"Rolling Shutter",
    pvc_consts.EXPOSE_OUT_LINE_TRIGGER:   "Line Trigger",
    pvc_consts.EXPOSE_OUT_GLOBAL_SHUTTER: "Global Shutter",
}

# --- Sensor clear modes (PARAM_CLEAR_MODE) ---
Clear_Mode = {
    pvc_consts.CLEAR_NEVER:                  "Never",
    pvc_consts.CLEAR_PRE_EXPOSURE:           "Pre-Exposure",
    pvc_consts.CLEAR_PRE_SEQUENCE:           "Pre-Sequence",
    pvc_consts.CLEAR_POST_SEQUENCE:          "Post-Sequence",
    pvc_consts.CLEAR_PRE_POST_SEQUENCE:      "Pre & Post Sequence",
    pvc_consts.CLEAR_PRE_EXPOSURE_POST_SEQ:  "Pre-Exposure + Post-Sequence",
}


class KinetixCamera:
    """
    Teledyne Photometrics Kinetix22 sCMOS camera.
    """

    class CameraSettings:
        def __init__(self):
            # --- Temperature ---
            self.temperature = None          # current, °C
            self.temp_setpoint = -10.0       # target, °C (Kinetix22 nominal: -10 °C air)

            # --- Exposure ---
            self.t_exposure = 10             # ms  (cam.exp_time unit)

            # --- ROI (in unbinned sensor pixels, 0-indexed) ---
            self.s1 = 0                      # serial (x) start
            self.s2 = 2399                   # serial (x) end
            self.p1 = 0                      # parallel (y) start
            self.p2 = 2399                   # parallel (y) end

            # --- Binning ---
            self.s_bin = 1                   # serial binning factor
            self.p_bin = 1                   # parallel binning factor

            # --- Derived image dimensions (updated by set_roi) ---
            self.pixels_x = 2400             # output image width  (serial)
            self.pixels_y = 2400             # output image height (parallel)
            self.ps = 6.5                    # effective pixel size, μm

            # --- Readout configuration ---
            # After any port change: re-apply speed, then gain (in that order).
            self.port_index = 0              # readout port index
            self.speed_index = 0             # speed table index within port
            self.gain_index = 1              # gain index within speed

            # --- Acquisition buffer ---
            self.buffer_frame_count = 16     # SDK circular buffer depth (frames)

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self._settings = self.CameraSettings()
        self.cam = None
        self.camera_name = 'PMPCIECam00'
        self._last_frame_count = -1
        self.data = None
        self.acq_thread = None
        self._initialize_sdk()
        if self.cam is not None:
            self.get_sn()

    def __getattr__(self, item):
        if '_settings' in self.__dict__ and hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def _initialize_sdk(self):
        try:
            pvc.init_pvcam()
            if self.camera_name:
                self.cam = Camera.select_camera(self.camera_name)
            else:
                self.cam = next(Camera.detect_camera())
            self.cam.open()
            self.logg.info(f"Kinetix camera opened: {self.cam.name}")
        except StopIteration:
            self.logg.error("No Kinetix camera detected by PVCAM")
            self.cam = None
        except Exception as e:
            self.logg.error(f"Error initializing Kinetix camera: {e}")
            self.cam = None

    def close(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()
            self.data = None
        try:
            if self.cam is not None and self.cam.is_open:
                self.cam.close()
                self.logg.info("Kinetix camera closed")
        except Exception as e:
            self.logg.error(f"Error closing Kinetix camera: {e}")
        self.cam = None  # drop reference before uninit so Camera.__del__ fires while SDK is still live
        try:
            pvc.uninit_pvcam()
        except Exception as e:
            self.logg.error(f"Error uninitializing PVCAM: {e}")

    def get_sn(self):
        try:
            sn = self.cam.serial_no
            self.logg.info(f"Kinetix Serial Number: {sn}")
        except Exception as e:
            self.logg.error(f"Error reading serial number: {e}")

    def get_temperature(self):
        """Read current sensor temperature (°C)."""
        try:
            self._settings.temperature = self.cam.temp  # already in °C
            self.logg.info(f"Kinetix temperature: {self._settings.temperature:.1f} °C")
        except Exception as e:
            self.logg.error(f"Error reading temperature: {e}")

    def check_camera_status(self):
        try:
            status = self.cam.check_frame_status()
            self.logg.info(f"Camera frame status: {status}")
        except Exception as e:
            self.logg.error(f"Error checking camera status: {e}")

    def set_temperature(self, setpoint_c=None):
        """Set cooling target. Default uses self.temp_setpoint."""
        if setpoint_c is not None:
            self._settings.temp_setpoint = float(setpoint_c)
        try:
            self.cam.temp_setpoint = self._settings.temp_setpoint
            self.logg.info(f"Temperature setpoint: {self._settings.temp_setpoint:.1f} °C")
        except Exception as e:
            self.logg.error(f"Error setting temperature: {e}")

    def set_readout_port(self, port_index=0):
        """
        Select the readout port. On the Kinetix22 the ports expose different
        readout modes (Sensitivity, Speed, Dynamic Range, Sub-Electron…).
        After changing the port, always re-apply speed then gain.
        """
        try:
            self._settings.port_index = port_index
            self.cam.readout_port = port_index
            self.logg.info(f"Readout port set to index {port_index}")
        except Exception as e:
            self.logg.error(f"Error setting readout port: {e}")

    def set_speed(self, speed_index=0):
        """Select readout speed within the current port."""
        try:
            self._settings.speed_index = speed_index
            self.cam.speed = speed_index
            self.logg.info(
                f"Speed index {speed_index}"
                + (f" ({self.cam.speed_name})" if hasattr(self.cam, 'speed_name') else "")
            )
        except Exception as e:
            self.logg.error(f"Error setting speed: {e}")

    def set_gain(self, gain_index=1):
        """Select gain index within the current speed table entry."""
        try:
            self._settings.gain_index = gain_index
            self.cam.gain = gain_index
            self.logg.info(
                f"Gain index {gain_index}"
                + (f" ({self.cam.gain_name})" if hasattr(self.cam, 'gain_name') else "")
            )
        except Exception as e:
            self.logg.error(f"Error setting gain: {e}")

    def set_readout_config(self, port=None, speed=None, gain=None):
        """
        Apply port/speed/gain in the correct order.
        Omitting a value leaves that setting unchanged.
        """
        if port is not None:
            self.set_readout_port(port)
        if speed is not None:
            self.set_speed(speed)
        if gain is not None:
            self.set_gain(gain)

    def set_roi(self):
        """
        Apply the ROI defined in settings (s1, p1, s2, p2 in unbinned pixels).
        Recalculates output image dimensions after binning.

        PVCAM rule: cam.set_roi() inherits binning from the current first ROI,
        so binning must be committed to the camera BEFORE calling set_roi().
        """
        try:
            self.cam.binning = (self._settings.s_bin, self._settings.p_bin)
            self.cam.set_roi(self._settings.s1, self._settings.p1, self._settings.pixels_x, self._settings.pixels_y)
            self._settings.ps = 6.5 * self._settings.s_bin
            self.logg.info(
                f"ROI: s1={self._settings.s1}"
                f"p1={self._settings.p1}"
                f"bin={self._settings.s_bin}x{self._settings.p_bin} "
                f"→ {self._settings.pixels_x}x{self._settings.pixels_y} px"
            )
        except Exception as e:
            self.logg.error(f"Error setting ROI: {e}")

    def set_trigger_mode(self, exp_mode, expose_out_mode=None):
        """
        Set the trigger and expose-out modes.

        exp_mode: one of the pvc_consts exposure mode values, e.g.:
          pvc_consts.TIMED_MODE          – internal free-running
          pvc_consts.STROBED_MODE        – external edge trigger
          pvc_consts.EXT_TRIG_EDGE_RISING – external rising-edge (alternative)
          pvc_consts.EXT_TRIG_LEVEL      – external level trigger

        expose_out_mode: optional output signal mode, e.g.:
          pvc_consts.EXPOSE_OUT_ALL_ROWS      – high while any row is exposing
          pvc_consts.EXPOSE_OUT_GLOBAL_SHUTTER – global shutter signal

        If expose_out_mode is None, only exp_mode is applied.
        """
        try:
            self.cam.exp_mode = exp_mode
            self.logg.info(f"Exposure mode: {Exposure_Mode.get(exp_mode, exp_mode)}")
        except Exception as e:
            self.logg.error(f"Error setting exposure mode: {e}")
        if expose_out_mode is not None:
            try:
                self.cam.exp_out_mode = expose_out_mode
                self.logg.info(f"Expose-out mode: {Expose_Out_Mode.get(expose_out_mode, expose_out_mode)}")
            except Exception as e:
                self.logg.error(f"Error setting expose-out mode: {e}")

    def set_exposure_time(self, t_ms=10):
        """Set exposure time in milliseconds."""
        if t_ms is not None:
            self._settings.t_exposure = int(t_ms)
        try:
            self.cam.exp_time = self._settings.t_exposure
            self.logg.info(f"Exposure time: {self._settings.t_exposure} ms")
        except Exception as e:
            self.logg.error(f"Error setting exposure time: {e}")

    def set_clear_mode(self, mode=pvc_consts.CLEAR_PRE_SEQUENCE):
        try:
            self.cam.clear_mode = mode
            self.logg.info(f"Clear mode: {Clear_Mode.get(mode, mode)}")
        except Exception as e:
            self.logg.error(f"Error setting clear mode: {e}")

    def get_acquisition_timings(self):
        """Log readout time and pixel time for the current configuration."""
        try:
            t_readout = self.cam.get_param(pvc_consts.PARAM_READOUT_TIME)
            self.logg.info(f"Readout time: {t_readout:.3f} ms")
        except Exception as e:
            self.logg.error(f"Error reading readout time: {e}")
        try:
            pix_time = self.cam.pix_time  # ns per pixel
            self.logg.info(f"Pixel time: {pix_time} ns")
        except Exception as e:
            self.logg.error(f"Error reading pixel time: {e}")

    def prepare_live(self, port=1, speed=0, gain=1,
                     exp_mode=1792, expose_out=0):
        """Configure camera for live / preview acquisition."""
        self.set_readout_port(port)
        self.set_speed(speed)
        self.set_gain(gain)
        self.set_roi()
        self.set_trigger_mode(exp_mode, expose_out)
        self.set_exposure_time()
        self.get_acquisition_timings()

    def start_live(self):
        """Start live acquisition and the polling thread."""
        self.data = run_threads.CameraDataList(max_length=self._settings.buffer_frame_count)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)
        self._last_frame_count = -1
        try:
            self.cam.start_live(
                exp_time=self._settings.t_exposure,
                buffer_frame_count=self._settings.buffer_frame_count,
            )
            self.acq_thread.start()
            self.logg.info("Kinetix live acquisition started")
        except Exception as e:
            self.logg.error(f"Error starting live acquisition: {e}")

    def stop_live(self):
        """Abort hardware acquisition first, then stop the polling thread."""
        try:
            self.cam.finish()
            self.logg.info("Kinetix live acquisition stopped")
        except Exception as e:
            self.logg.error(f"Error stopping live acquisition: {e}")
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        self.data = None

    def get_images(self):
        """
        Poll the SDK for the next available frame and push it into self.data.
        Skips duplicate frames using the SDK frame counter.
        Called from CameraAcquisitionThread.run() under the thread lock.
        """
        if self.data is None:
            return
        try:
            # timeout_ms=50: don't block the thread longer than one poll period
            frame_dict, fps, frame_count = self.cam.poll_frame(
                timeout_ms=50, copyData=True
            )
        except Exception:
            return
        if frame_dict is None:
            return
        pixel_data = frame_dict.get('pixel_data')
        if pixel_data is None:
            return
        if frame_count == self._last_frame_count:
            return  # no new frame since last poll
        self._last_frame_count = frame_count
        if self.data is None:
            return  # was cleared by stop_live() while we were polling
        self.data.add_element([pixel_data], [frame_count, frame_count])

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element()
        return None

    def prepare_data_acquisition(self, port=0, speed=0, gain=1,
                                 exp_mode=pvc_consts.STROBED_MODE,
                                 expose_out=pvc_consts.EXPOSE_OUT_ALL_ROWS):
        """Configure camera for triggered data acquisition."""
        self.set_readout_port(port)
        self.set_speed(speed)
        self.set_gain(gain)
        self.set_roi()
        self.set_trigger_mode(exp_mode, expose_out)
        self.set_exposure_time()
        self.get_acquisition_timings()

    def start_data_acquisition(self, n, fd, fn):
        """
        Start triggered acquisition saving n frames per TIFF stack.

        n  : number of frames per saved stack
        fd : save directory path
        fn : file name prefix
        """
        self.data = run_threads.CameraDataList(
            max_length=n, save_to_disk=True, save_dir=fd, file_prefix=fn
        )
        self.acq_thread = run_threads.CameraAcquisitionThread(self)
        self._last_frame_count = -1
        try:
            self.cam.start_live(
                exp_time=self._settings.t_exposure,
                buffer_frame_count=self._settings.buffer_frame_count,
            )
            self.acq_thread.start()
            self.logg.info("Kinetix data acquisition started")
        except Exception as e:
            self.logg.error(f"Error starting data acquisition: {e}")

    def stop_data_acquisition(self):
        """Abort hardware first, then drain the TIFF-writer queue."""
        try:
            self.cam.finish()
            self.logg.info("Kinetix data acquisition stopped")
        except Exception as e:
            self.logg.error(f"Error stopping data acquisition: {e}")
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        if self.data is not None:
            self.data.close()   # flush + join the background TIFF-writer thread
            self.data = None

    def get_data(self):
        if self.data is not None:
            return self.data.get_elements()
        return None
