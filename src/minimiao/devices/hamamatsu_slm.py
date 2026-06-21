# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import atexit
import numpy as np
from PIL import Image
import ctypes as ct
from ctypes import c_int32, c_uint8, c_uint32, c_double, c_char, POINTER, create_string_buffer

from minimiao import logger

lib_path = r"C:\Users\Janos\OneDrive\Asztali gép\Hamamatsu\USB_Control_SDK\hpkSLMdaLV_stdcall_64bit\hpkSLMdaLV.dll"

class HamamatsuSLM:

    def __init__(self, serial_number="LSH0805629", logg=None):
        self.logg = logg or logger.setup_logging()
        self.lib = None
        self.bid = None
        if lib_path is not None:
            self.lib = ct.CDLL(lib_path)
            self._bind_functions()
            atexit.register(self._unload_dll)
            if lib_path is not None:
                self.serial_number = serial_number
                self.bid = self.open()
                if self.bid is not None:
                    head_temp, cb_temp = self.check_temp()
                    self.logg.info(f"Head Temperature: {head_temp}, Controller Temperature: {cb_temp}")
                    self.nx = 1272
                    self.ny = 1024
                    self.array_size = self.nx * self.ny
                    self.pattern = np.zeros((self.nx, self.ny), dtype=np.uint8)
                else:
                    self.logg.error(f"No devices found")
            else:
                self.logg.error("No serial number specified")
        else:
            self.logg.error("No lib path specified")

    def _bind_functions(self):
        # Open_Dev
        self.lib.Open_Dev.argtypes = [POINTER(c_uint8), c_int32]
        self.lib.Open_Dev.restype = c_int32

        # Close_Dev
        self.lib.Close_Dev.argtypes = [POINTER(c_uint8), c_int32]
        self.lib.Close_Dev.restype = c_int32

        # Check_HeadSerial
        self.lib.Check_HeadSerial.argtypes = [c_uint8, POINTER(c_char), c_int32]
        self.lib.Check_HeadSerial.restype = c_int32

        # Write_FMemArray
        self.lib.Write_FMemArray.argtypes = [c_uint8, POINTER(c_uint8), c_int32, c_uint32, c_uint32, c_uint32]
        self.lib.Write_FMemArray.restype = c_int32

        # Write_FMemBMPPath
        self.lib.Write_FMemBMPPath.argtypes = [c_uint8, POINTER(c_char), c_int32]
        self.lib.Write_FMemBMPPath.restype = c_int32

        # Check_Temp
        self.lib.Check_Temp.argtypes = [c_uint8, POINTER(c_double), POINTER(c_double)]
        self.lib.Check_Temp.restype = c_int32

        # Mode_Select
        self.lib.Mode_Select.argtypes = [c_uint8, c_uint8]
        self.lib.Mode_Select.restype = c_int32

        # Mode_Check
        self.lib.Mode_Check.argtypes = [c_uint8, POINTER(c_uint32)]
        self.lib.Mode_Check.restype = c_int32

        # Change_DispSlot
        self.lib.Change_DispSlot.argtypes = [c_uint8, c_int32]
        self.lib.Change_DispSlot.restype = c_int32

    def open(self):
        size = 8
        bids = (c_uint8 * size)()
        dvs = self.lib.Open_Dev(bids, size)
        if dvs <= 0:
            self.logg.error(f"No devices found")
            return None
        else:
            for i in range(dvs):
                bid = bids[i]
                ser = self.get_serial(bid)
                if ser is None:
                    continue
                if ser == self.serial_number:
                    self.logg.info(f"Connected to SLM: {ser}")
                    return bid
            self.logg.error(f"SLM Serial: '{self.serial_number}' not found")
            return None

    def close(self):
        if self.bid is not None:
            bid_list = (c_uint8 * 1)(self.bid)
            ret = self.lib.Close_Dev(bid_list, 1)
            self.bid = None
            if ret == 1:
                self.logg.info(f"SLM Closed")
            else:
                self.logg.error(f"Error when closing SLM")
        self._unload_dll()

    def _unload_dll(self):
        # Explicitly FreeLibrary the SLM DLL so its DllMain DLL_PROCESS_DETACH
        # runs now (while PVCAM is still live), not later during OS process exit
        if self.lib is not None and self.lib._handle:
            handle = self.lib._handle
            self.lib._handle = 0  # prevent ctypes double-free
            ct.windll.kernel32.FreeLibrary(handle)

    def get_serial(self, bid, char_size=11):
        buf = create_string_buffer(char_size)
        ret = self.lib.Check_HeadSerial(bid, buf, char_size)
        if ret:
            return buf.value.decode("ascii", errors="ignore")
        else:
            return None

    def read_pattern(self, file):
        pattern = Image.open(file)
        self.pattern = np.array(pattern, dtype=np.uint8)

    def load_pattern(self, pfd, slot_no=0):
        self.read_pattern(pfd)
        self.display_pattern(slot_no)
        array = self.pattern.flatten()
        if len(array) != self.array_size:
            raise ValueError(
                f"Array length {len(array)} does not match expected size {self.array_size})"
            )
        c_array = (c_uint8 * self.array_size)(*array)
        ret = self.lib.Write_FMemArray(self.bid, c_array, self.array_size, self.nx, self.ny, slot_no)
        if ret == 1:
            self.logg.info(f"Pattern Loaded to Slot: {slot_no}")
        else:
            self.logg.error(f"Failed to load pattern")

    def display_pattern(self, slot_no=0):
        ret = self.lib.Change_DispSlot(self.bid, slot_no)
        if ret == 1:
            self.logg.info(f"Display Slot: {slot_no}")
        else:
            self.logg.error(f"Failed to display Slot: {slot_no}")

    def check_temp(self):
        head_temp = c_double()
        cb_temp = c_double()
        ret = self.lib.Check_Temp(self.bid, ct.byref(head_temp), ct.byref(cb_temp))
        if ret == 1:
            return head_temp.value, cb_temp.value
        return None

    def mode_select(self, mode):
        """
        Set operation mode.

        Parameters
        ----------
        mode : int (0 = DVI, 1 = USB/Trigger)

        Returns
        -------
        bool
        """
        ret = self.lib.Mode_Select(self.bid, mode)
        return ret == 1

    def mode_check(self, bid):
        """
        Get current operation mode.

        Returns
        -------
        int or None
            0 = DVI, 1 = USB/Trigger, None = failed
        """
        mode = c_uint32()
        ret = self.lib.Mode_Check(bid, ct.byref(mode))
        if ret:
            return mode.value
        return None
