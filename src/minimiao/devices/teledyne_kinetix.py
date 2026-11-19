# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import time
from collections import deque

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
            self.t_exposure = 10.0           # ms (applied with 1 us resolution when the camera supports it)

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
        self.recorder = None
        self.trigger_mode = "internal"
        self._exp_us = False        # exposure resolution in use: True = microseconds, False = milliseconds
        self._sequence = False      # current recording is a fixed-length sequence
        self._fps_samples = deque()  # (monotonic time, SDK frame counter) over the last ~1 s
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
            self.cam.reset_rois()
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

    def _set_exposure_resolution(self):
        """Use 1 us exposure resolution if the camera accepts it, else 1 ms."""
        us = getattr(pvc_consts, "EXP_RES_ONE_MICROSEC", 1)
        try:
            self.cam.exp_res = us
            self._exp_us = True
        except Exception as e:
            if self._exp_us or not hasattr(self, "_exp_res_warned"):
                self.logg.warning(f"Microsecond exposure resolution not available ({e}); using 1 ms steps")
                self._exp_res_warned = True
            self._exp_us = False

    def _exposure_units(self):
        """Exposure in the camera's current resolution units (what exp_time / start_* expect)."""
        ms = float(self._settings.t_exposure)
        return max(1, int(round(ms * 1000))) if self._exp_us else max(1, int(round(ms)))

    def set_exposure_time(self, t_ms=None):
        """Set exposure time in ms (float; None = use settings value). Applied with 1 us
        resolution when supported, otherwise rounded to whole ms; the applied value is
        stored back in settings."""
        if t_ms is not None:
            self._settings.t_exposure = float(t_ms)
        self._set_exposure_resolution()
        units = self._exposure_units()
        self._settings.t_exposure = units / 1000.0 if self._exp_us else float(units)
        try:
            self.cam.exp_time = units
            self.logg.info(f"Exposure time: {self._settings.t_exposure:g} ms "
                           f"({'1 us' if self._exp_us else '1 ms'} resolution)")
        except Exception as e:
            self.logg.error(f"Error setting exposure time: {e}")

    def get_readout_options(self):
        """
        Readout modes reported by the camera: list of dicts
        {port, port_name, speed, pixel_time_ns, bit_depth, gain, gain_name}
        (one entry per port/speed with its first gain). Empty list if unavailable.
        """
        options = []
        try:
            table = self.cam.port_speed_gain_table
        except Exception as e:
            self.logg.warning(f"Readout table not available: {e}")
            return options
        try:
            for port_name, port in table.items():
                port_value = port.get("port_value")
                for skey, speed in port.items():
                    if not isinstance(speed, dict) or "speed_index" not in speed:
                        continue
                    gains = [(gname, g.get("gain_index")) for gname, g in speed.items()
                             if isinstance(g, dict) and "gain_index" in g]
                    gname, gidx = gains[0] if gains else (None, 1)
                    options.append({"port": int(port_value), "port_name": str(port_name),
                                    "speed": int(speed["speed_index"]),
                                    "pixel_time_ns": speed.get("pixel_time"),
                                    "bit_depth": speed.get("bit_depth"),
                                    "gain": int(gidx) if gidx is not None else 1,
                                    "gain_name": gname})
        except Exception as e:
            self.logg.warning(f"Could not parse readout table: {e}")
        return options

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

    def prepare_live(self, port=0, speed=0, gain=1,
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
        self._fps_samples.clear()
        self._last_frame_count = -1
        try:
            self.cam.start_live(
                exp_time=self._exposure_units(),
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
        Called repeatedly from CameraAcquisitionThread.run().
        Returns True if a new frame was added, False otherwise.
        """
        if self.data is None:
            return False
        try:
            # timeout_ms=50: don't block the thread longer than one poll period
            frame_dict, fps, frame_count = self.cam.poll_frame(
                timeout_ms=50, copyData=True
            )
        except Exception:
            return False
        if frame_dict is None:
            return False
        pixel_data = frame_dict.get('pixel_data')
        if pixel_data is None:
            return False
        if frame_count == self._last_frame_count:
            return False  # no new frame since last poll
        self._last_frame_count = frame_count
        self._note_frame(frame_count)
        data = self.data
        if data is None:
            return False  # was cleared by stop_live() while we were polling
        data.add_element([pixel_data], [frame_count, frame_count])
        return True

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element()
        return None

    # ---- measured frame rate ----------------------------------------------------

    FPS_WINDOW_S = 1.0

    def _note_frame(self, frame_count):
        """Record a received frame's SDK counter; called from the thread that reads frames."""
        now = time.monotonic()
        s = self._fps_samples
        s.append((now, frame_count))
        while len(s) > 2 and now - s[0][0] > self.FPS_WINDOW_S:
            s.popleft()

    @staticmethod
    def clock():
        """This (camera) process's perf_counter: the clock preview-frame read times are stamped with."""
        return time.perf_counter()

    def current_fps(self):
        """Camera frame rate over the last ~1 s from the SDK frame counter (counts every frame
        the camera took, including any that were dropped), or 0.0 if unknown."""
        s = self._fps_samples
        try:
            (t0, c0), (t1, c1) = s[0], s[-1]
        except IndexError:
            return 0.0
        if t1 - t0 <= 0 or c0 is None or c1 is None:
            return 0.0
        return max(0.0, (c1 - c0) / (t1 - t0))

    # ---- lossless single-stack recording ---------------------------------------

    TRIGGER_MODES = {"internal": pvc_consts.EXT_TRIG_INTERNAL,
                     "external": pvc_consts.EXT_TRIG_EDGE_RISING}

    def prepare_recording(self, trigger="internal", port=1, speed=0, gain=1, expose_out=0):
        """
        Configure for recording. trigger: 'internal' (free-running, frame rate
        set by exposure/readout) or 'external' (one frame per rising edge on
        the trigger input).
        """
        if trigger not in self.TRIGGER_MODES:
            raise ValueError(f"Unknown trigger mode {trigger!r}")
        self.set_readout_port(port)
        self.set_speed(speed)
        self.set_gain(gain)
        self.set_roi()
        self.set_trigger_mode(self.TRIGGER_MODES[trigger], expose_out)
        self.set_exposure_time()
        self.trigger_mode = trigger
        self.get_acquisition_timings()

    def get_recording_info(self):
        """Settings actually applied, for metadata."""
        info = {"exposure_ms": self._settings.t_exposure, "exposure_resolution": "1 us" if self._exp_us else "1 ms",
                "trigger_mode": self.trigger_mode,
                "s1": self._settings.s1, "p1": self._settings.p1,
                "pixels_x": self._settings.pixels_x, "pixels_y": self._settings.pixels_y,
                "s_bin": self._settings.s_bin, "p_bin": self._settings.p_bin,
                "port_index": self._settings.port_index, "speed_index": self._settings.speed_index,
                "gain_index": self._settings.gain_index}
        for key, getter in (("readout_time_ms", lambda: self.cam.get_param(pvc_consts.PARAM_READOUT_TIME)),
                            ("serial_no", lambda: self.cam.serial_no),
                            ("temperature_c", lambda: self.cam.temp),
                            ("speed_name", lambda: self.cam.speed_name),
                            ("gain_name", lambda: self.cam.gain_name)):
            try:
                info[key] = getter()
            except Exception:
                info[key] = None
        return info

    def _poll_one(self, timeout_ms=50):
        """Oldest unread frame from the SDK buffer as (pixel_data, frame_nr), or None."""
        try:
            frame_dict, fps, frame_count = self.cam.poll_frame(timeout_ms=timeout_ms, oldestFrame=True,
                                                               copyData=True)
        except Exception:
            return None
        if frame_dict is None or frame_dict.get('pixel_data') is None:
            return None
        self._note_frame(frame_count)
        return frame_dict['pixel_data'], frame_count

    MAX_SEQUENCE_BYTES = 8 * 2 ** 30  # PVCAM holds a whole sequence in RAM

    def start_recording(self, path, n_frames=None, buffer_mb=2048, bigtiff=False, preview=None,
                        frames_per_file=None):
        """
        Record into one TIFF at `path`.
        n_frames > 0: fixed-length sequence (start_seq). PVCAM buffers all n frames, so a
                      slow reader delays frames but cannot lose them; stops by itself.
        n_frames None/0: continuous (start_live, ring buffer of up to 1024 frames) until stop.
        frames_per_file > 0 (continuous only): a new TIFF every that many frames
                      (<stem>_00000.tif, _00001.tif, ...).
        """
        frame_bytes = max(1, self._settings.pixels_x * self._settings.pixels_y * 2)
        units = self._exposure_units()
        self._fps_samples.clear()
        if n_frames:
            n_frames = int(n_frames)
            if n_frames * frame_bytes > self.MAX_SEQUENCE_BYTES:
                raise ValueError(f"Sequence of {n_frames} frames needs {n_frames * frame_bytes / 2 ** 30:.1f} GB "
                                 f"of camera buffer (limit {self.MAX_SEQUENCE_BYTES / 2 ** 30:.0f} GB); "
                                 f"use fewer frames, a smaller ROI, or continuous recording")
            self.recorder = run_threads.StackRecorder(self._poll_one, path, bigtiff=bigtiff, preview=preview,
                                                      max_frames=n_frames)
            self._sequence = True
            self.cam.start_seq(exp_time=units, num_frames=n_frames)
            mode = f"sequence of {n_frames} frames"
        else:
            self._settings.buffer_frame_count = int(min(1024, max(16, buffer_mb * 2 ** 20 // frame_bytes)))
            self.recorder = run_threads.StackRecorder(self._poll_one, path, bigtiff=bigtiff, preview=preview,
                                                      frames_per_file=frames_per_file)
            self._sequence = False
            self.cam.start_live(exp_time=units, buffer_frame_count=self._settings.buffer_frame_count)
            mode = (f"continuous, buffer {self._settings.buffer_frame_count} frames"
                    + (f", a new file every {frames_per_file} frames" if frames_per_file else ""))
        self.recorder.start()
        self.logg.info(f"Recording to {path} ({mode}; up to {self.recorder.max_queue_bytes / 2 ** 30:.1f} GB "
                       f"of frames can wait in RAM for the disk)")

    def recording_progress(self):
        """{'n_frames': frames received so far, 'done': fixed-length sequence complete}"""
        if self.recorder is None:
            return {"n_frames": 0, "done": True, "active": False}
        return {"n_frames": self.recorder.n_frames, "done": self.recorder.is_done(), "active": True}

    def stop_recording(self, wait_s=0.0):
        """
        Stop recording and close the TIFF; returns the recording summary.
        wait_s: for a sequence, wait up to this long for the remaining frames first.
        """
        if self.recorder is None:
            return None
        if self._sequence and wait_s and wait_s > 0:
            if not self.recorder.wait_done(wait_s):
                self.logg.warning(f"Sequence incomplete after waiting {wait_s:.1f} s: "
                                  f"{self.recorder.n_frames}/{self.recorder.max_frames} frames")
        summary = self.recorder.stop()
        self.recorder = None
        try:
            self.cam.finish()
        except Exception as e:
            self.logg.error(f"Error stopping recording: {e}")
        summary["mode"] = "sequence" if self._sequence else "continuous"
        summary["buffer_frame_count"] = None if self._sequence else self._settings.buffer_frame_count
        self._sequence = False
        self.logg.info(f"Recording stopped: {summary['n_frames']} frames"
                       + (f" of {summary['n_frames_requested']}" if summary.get("n_frames_requested") else "")
                       + f", {summary['n_dropped']} dropped")
        return summary
