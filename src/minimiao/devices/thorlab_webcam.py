import numpy as np
import ctypes as ct

uc480_dll = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\uc480_64.dll"


class ThorCam:
    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.uc480 = self._init_dll()
        if self.uc480 is not None:
            try:
                self.handle = self._init_cam()
            except Exception as e:
                self.handle = None
                self.logg.error(f"{e}")
                return
        if self.handle is not None:
            self._config_cam()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _init_dll(self):
        try:
            dll = ct.windll.LoadLibrary(uc480_dll)
            return dll
        except Exception as e:
            self.logg.error(f"ThorCam drivers not available {e}")
            return None

    def _init_cam(self):
        is_init_camera = self.uc480.is_InitCamera
        is_init_camera.argtypes = [ct.POINTER(ct.c_int)]
        h = ct.c_int(0)
        i = is_init_camera(ct.byref(h))
        if i == 0:
            self.logg.info(f"ThorCam initiated successfully")
            return h
        else:
            raise RuntimeError(f"Opening the ThorCam failed with error code {i}")

    def _config_cam(self):
        try:
            self.roi_shape = self.set_roi_shape((1024, 1024))
        except Exception as e:
            self.logg.error(f"{e}")
            return
        try:
            self.roi_pos = self.set_roi_pos((0, 0))
        except Exception as e:
            self.logg.error(f"{e}")
            return
        self.meminfo = None
        self.meminfo = self.initialize_memory()
        pixel_clock = ct.c_uint(5)  # set pixel clock to 5 MHz
        is_pixel_clock = self.uc480.is_PixelClock
        is_pixel_clock.argtypes = [ct.c_int, ct.c_uint, ct.POINTER(ct.c_uint), ct.c_uint]
        is_pixel_clock(self.handle, 6, ct.byref(pixel_clock), ct.sizeof(pixel_clock))  # 6 for setting pixel clock
        self.uc480.is_SetColorMode(self.handle, 6)  # 6 is for monochrome 8 bit. See uc480.h for definitions
        self.frame_rate = self.set_frame_rate(4)
        self.exposure = self.set_exposure(32)

    def close(self):
        if self.handle is not None:
            self.stop_live()
            i = self.uc480.is_ExitCamera(self.handle)
            if i == 0:
                self.logg.info("ThorCam closed successfully.")
            else:
                self.logg.error("Closing ThorCam failed with error code " + str(i))
        else:
            return

    def set_roi_shape(self, set_roi_shape):
        class IS_SIZE_2D(ct.Structure):
            _fields_ = [('s32Width', ct.c_int), ('s32Height', ct.c_int)]

        aoi_size = IS_SIZE_2D(set_roi_shape[0], set_roi_shape[1])  # Width and Height

        is_aoi = self.uc480.is_AOI
        is_aoi.argtypes = [ct.c_int, ct.c_uint, ct.POINTER(IS_SIZE_2D), ct.c_uint]
        i = is_aoi(self.handle, 5, ct.byref(aoi_size), 8)  # 5 for setting size, 3 for setting position
        if i == 0:
            self.logg.info(f"ThorCam ROI size set successfully.")
            is_aoi(self.handle, 6, ct.byref(aoi_size), 8)  # 6 for getting size, 4 for getting position
            return [aoi_size.s32Width, aoi_size.s32Height]
        else:
            raise RuntimeError(f"Set ThorCam ROI size failed with error code {i}")

    def set_roi_pos(self, set_roi_pos):
        class IS_POINT_2D(ct.Structure):
            _fields_ = [('s32X', ct.c_int), ('s32Y', ct.c_int)]

        aoi_pos = IS_POINT_2D(set_roi_pos[0], set_roi_pos[1])  # Width and Height

        is_aoi = self.uc480.is_AOI
        is_aoi.argtypes = [ct.c_int, ct.c_uint, ct.POINTER(IS_POINT_2D), ct.c_uint]
        i = is_aoi(self.handle, 3, ct.byref(aoi_pos), 8)  # 5 for setting size, 3 for setting position
        if i == 0:
            self.logg.info(f"ThorCam ROI position set successfully.")
            is_aoi(self.handle, 4, ct.byref(aoi_pos), 8)  # 6 for getting size, 4 for getting position
            return [aoi_pos.s32X, aoi_pos.s32Y]
        else:
            raise RuntimeError(f"Set ThorCam ROI size failed with error code {i}")

    def initialize_memory(self):
        if self.meminfo is not None:
            self.uc480.is_FreeImageMem(self.handle, self.meminfo[0], self.meminfo[1])
        dim_x = self.roi_shape[0]
        dim_y = self.roi_shape[1]
        image_size = dim_x * dim_y
        memid = ct.c_int(0)
        c_buf = (ct.c_ubyte * image_size)(0)
        self.uc480.is_SetAllocatedImageMem(self.handle, dim_x, dim_y, 8, c_buf, ct.byref(memid))
        self.uc480.is_SetImageMem(self.handle, c_buf, memid)
        return [c_buf, memid]

    def set_frame_rate(self, frame_rate):
        is_set_frame_rate = self.uc480.is_SetFrameRate
        set_frame_rate = ct.c_double(0)
        is_set_frame_rate.argtypes = [ct.c_int, ct.c_double, ct.POINTER(ct.c_double)]
        is_set_frame_rate(self.handle, frame_rate, ct.byref(set_frame_rate))
        return set_frame_rate.value

    def set_exposure(self, exposure):
        """ exposure in ms """
        exposure_c = ct.c_double(exposure)
        is_exposure = self.uc480.is_Exposure
        is_exposure.argtypes = [ct.c_int, ct.c_uint, ct.POINTER(ct.c_double), ct.c_uint]
        is_exposure(self.handle, 12, exposure_c, 8)  # 12 is for setting exposure
        return exposure_c.value

    def start_live(self):
        self.uc480.is_CaptureVideo(self.handle, 1)

    def stop_live(self):
        self.uc480.is_StopLiveVideo(self.handle, 1)

    def get_image(self):
        return np.frombuffer(self.meminfo[0], ct.c_ubyte).reshape(self.roi_shape[1], self.roi_shape[0])
