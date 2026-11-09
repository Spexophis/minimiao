# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.
"""
Thorlabs LCC25D dual channel liquid crystal controller (liquid crystal phase retarder), via the Thorlabs
LCC25D_COMMAND_LIB python wrapper.

The wrapper (LCC25D_COMMAND_LIB.py) and its dll (LCC25D_COMMAND_LIB_x64.dll / _x86.dll) are imported lazily, so this
module can be imported on a machine without the SDK; the constructor raises and the caller can carry on without the
retarder.

Channels are 0-based (0 = channel 1, 1 = channel 2). Each channel has four voltages (index 0..3 = Voltage 1..4):
Voltage 1 / Voltage 2 are the two retardance states of the output modes, Voltage 3 / Voltage 4 belong to the fast
switching mode.

Output modes: "modulation" (Voltage 1 and 2 alternate at the modulation frequency), "v1" (Voltage 1 only), "v2"
(Voltage 2 only) and "sequence" (the list set by set_sequence()).

Quick use:
    ret = LCC25DRetarder(logg=logg)
    ret.set_voltage(0, 0, 5.0)         # channel 1, Voltage 1 = 5 V
    ret.set_voltage(0, 1, 12.0)        # channel 1, Voltage 2 = 12 V
    ret.set_output_mode(0, "v1")
    ret.set_output(0, True)
    ret.set_output_mode(0, "v2")       # switch the retardance
    ret.close()
"""

import importlib
import os
import struct
import sys
import time

from minimiao import logger

LCC25D_DIR = r"C:\Program Files (x86)\Thorlabs\LCC25D\Documents\Thorlabs_LCC25D_PythonSDK"

MAX_VOLTAGE = 25.0  # V
MIN_MODULATION_HZ, MAX_MODULATION_HZ = 0.5, 150.0
MIN_SEQUENCE_MS, MAX_SEQUENCE_MS = 1, 5000
ANALOG_IN_MAX_V = 5.0  # V, ANLG IN accepts 0 - 5 V DC (above 5 V can damage the input); 5 V gives 25 V RMS
MIN_TEMPERATURE, MAX_TEMPERATURE = 10.0, 50.0  # C, user guide specification (the SDK docstring says 0 .. 50)
MAX_SEQUENCE_STEPS = 512

OUTPUT_MODES = {"modulation": 0, "v1": 1, "v2": 2, "sequence": 3}
CHANNEL_STATES = {1: "Normal", 2: "Warning", 3: "Error"}


class LCC25DRetarder:
    def __init__(self, logg=None, sn=None):
        """
        sn: serial number of the USB device to open (None: the first one found).
        """
        self.logg = logg or logger.setup_logging()
        self.lib_dir = LCC25D_DIR
        self.hdl = -1
        self.sn = None
        self._lcc = None
        self._load_sdk()
        self._lcc = self._sdk.LCC25D()
        try:
            self._open_usb(sn)
        except Exception:
            self._release()
            raise
        self.get_idn()

    # ------------------------------------------------------------------ SDK plumbing
    def _load_sdk(self):
        if self.lib_dir not in sys.path:
            sys.path.append(self.lib_dir)
        try:
            self._sdk = importlib.import_module("LCC25D_COMMAND_LIB")
        except ImportError as e:
            raise RuntimeError(f"LCC25D SDK wrapper not found in {self.lib_dir}: {e}")
        if not self._sdk.LCC25D.isLoad:
            # the wrapper's own default is a relative path to the 32-bit dll; pick the one that matches python
            bits = "x64" if struct.calcsize("P") == 8 else "x86"
            self._sdk.LCC25D.load_library(os.path.join(self.lib_dir, f"LCC25D_COMMAND_LIB_{bits}.dll"))

    def _open_usb(self, sn):
        devices = self._sdk.LCC25D.list_usb()  # [[serial number, COM port], ...]
        if not devices:
            raise RuntimeError("LCC25D not found: check that the controller is plugged in and its driver is installed")
        self.logg.info(f"LCC25D USB devices: {devices}")
        if sn is None:
            sn = devices[0][0]
        elif sn not in [d[0] for d in devices]:
            raise RuntimeError(f"LCC25D {sn} not found among {[d[0] for d in devices]}")
        self.hdl = self._lcc.open_usb(sn)
        self.sn = sn

    def _open_ethernet(self, ip, key, timeout_ms=10000):
        self.hdl = self._lcc.open_ethernet(ip, key, timeout_ms)
        self.sn = ip

    def _release(self):
        try:
            self._sdk.LCC25D.device_release()
        except Exception as e:
            self.logg.error(f"LCC25D release: {e}")

    @staticmethod
    def _check_channel(channel):
        if channel not in (0, 1):
            raise ValueError(f"channel must be 0 or 1 (channel 1 / 2), got {channel}")

    @staticmethod
    def _check_range(name, value, lo, hi):
        if not lo <= value <= hi:
            raise ValueError(f"{name} {value} is outside {lo} .. {hi}")

    def close(self):
        """Close the connection. The outputs are left as they are; use set_output(ch, False) before if wanted."""
        try:
            self.set_output(0, False)  # channel 1 output off (add channel 1 if you use it)
            # self.set_heating(0, False)  # only if you ever turned heating on; not needed for your LCC1613-A
            if self.hdl >= 0:
                self._lcc.close()
                self.hdl = -1
        finally:
            self._release()
        self.logg.info("LCC25D closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ device info
    def get_idn(self):
        """Identification string: manufacturer, device name, serial number, firmware version."""
        out = [""]
        self._lcc.get_IDN(out, 1024)
        self.logg.info(f"LCC25D: {out[0]}")
        return out[0]

    def get_channel_state(self, channel):
        """'Normal', 'Warning' or 'Error'."""
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_state(channel, out)
        return CHANNEL_STATES.get(out[0], str(out[0]))

    # ------------------------------------------------------------------ voltages
    def set_voltage(self, channel, index, voltage):
        """Voltage `index` (0..3 = Voltage 1..4) of a channel, 0 .. 25 V."""
        self._check_channel(channel)
        self._check_range("index", index, 0, 3)
        self._check_range("voltage", voltage, 0.0, MAX_VOLTAGE)
        self._lcc.set_channel_voltage(channel, index, float(voltage))
        self.logg.info(f"LCC25D channel {channel + 1} Voltage {index + 1} set to {voltage:g} V")

    def get_voltage(self, channel, index):
        self._check_channel(channel)
        self._check_range("index", index, 0, 3)
        out = [0.0]
        self._lcc.get_channel_voltage(channel, index, out)
        return out[0]

    # ------------------------------------------------------------------ output mode and state
    def set_output_mode(self, channel, mode):
        """mode: 'modulation', 'v1', 'v2', 'sequence' (or 0..3)."""
        self._check_channel(channel)
        code = OUTPUT_MODES[mode.lower()] if isinstance(mode, str) else int(mode)
        self._check_range("mode", code, 0, 3)
        self._lcc.set_channel_output_mode(channel, code)
        self.logg.info(f"LCC25D channel {channel + 1} output mode {mode}")

    def get_output_mode(self, channel):
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_output_mode(channel, out)
        return {v: k for k, v in OUTPUT_MODES.items()}.get(out[0], out[0])

    def set_output(self, channel, on):
        """Switch the channel output on or off."""
        self._check_channel(channel)
        self._lcc.set_channel_output_state(channel, 1 if on else 0)
        self.logg.info(f"LCC25D channel {channel + 1} output {'on' if on else 'off'}")

    def get_output(self, channel):
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_output_state(channel, out)
        return bool(out[0])

    def select_state(self, channel, state):
        """Switch a running channel between its two retardance states: state 1 -> Voltage 1, 2 -> Voltage 2."""
        if state not in (1, 2):
            raise ValueError("state must be 1 or 2")
        self.set_output_mode(channel, "v1" if state == 1 else "v2")

    # ------------------------------------------------------------------ modulation
    def set_modulation_frequency(self, channel, frequency):
        """Frequency at which 'modulation' mode alternates Voltage 1 / 2, 0.5 .. 150 Hz."""
        self._check_channel(channel)
        self._check_range("modulation frequency", frequency, MIN_MODULATION_HZ, MAX_MODULATION_HZ)
        self._lcc.set_channel_modulation_frequency(channel, float(frequency))
        self.logg.info(f"LCC25D channel {channel + 1} modulation frequency {frequency:g} Hz")

    def get_modulation_frequency(self, channel):
        self._check_channel(channel)
        out = [0.0]
        self._lcc.get_channel_modulation_frequency(channel, out)
        return out[0]

    def set_base_frequency(self, channel, frequency):
        """Frequency of the liquid crystal drive wave, 500 .. 10000 Hz."""
        self._check_channel(channel)
        self._check_range("base frequency", frequency, 500, 10000)
        self._lcc.set_channel_base_frequency(channel, int(frequency))

    def get_base_frequency(self, channel):
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_base_frequency(channel, out)
        return out[0]

    # ------------------------------------------------------------------ external control
    def set_trigger_in(self, channel, on):
        """
        Let a 5 V TTL pulse on TRIG IN switch the channel between Voltage 1 and 2. Only valid in "modulation" (SWITCH)
        mode; it excludes fast switching (enabling one disables the other on the controller).
        """
        self._check_channel(channel)
        self._lcc.set_channel_trigger_in_state(channel, 1 if on else 0)

    def set_trigger_out(self, channel, on):
        """TRIG OUT follows the state (high = Voltage 2, low = Voltage 1). Only valid in "modulation" (SWITCH) mode."""
        self._check_channel(channel)
        self._lcc.set_channel_trigger_out_state(channel, 1 if on else 0)

    def set_analog_in(self, channel, on):
        """ANALG IN (0 - 5 V) sets the voltage, 0 - 25 V. Only valid in "v1" / "v2" mode; it excludes fast switching."""
        self._check_channel(channel)
        self._lcc.set_channel_analog_in_state(channel, 1 if on else 0)

    def get_analog_in(self, channel):
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_analog_in_state(channel, out)
        return bool(out[0])

    @staticmethod
    def analog_volts_for_rms(rms):
        """Signal to put on ANLG IN for an LC drive of `rms` volts RMS: 0 - 5 V DC maps linearly to 0 - 25 V RMS."""
        if not 0.0 <= rms <= MAX_VOLTAGE:
            raise ValueError(f"rms {rms} is outside 0.0 .. {MAX_VOLTAGE}")
        return rms * ANALOG_IN_MAX_V / MAX_VOLTAGE

    @staticmethod
    def rms_for_analog_volts(volts):
        if not 0.0 <= volts <= ANALOG_IN_MAX_V:
            raise ValueError(f"analog voltage {volts} is outside 0.0 .. {ANALOG_IN_MAX_V}")
        return volts * MAX_VOLTAGE / ANALOG_IN_MAX_V

    def enable_analog_control(self, channel, mode="v1"):
        """
        Let the voltage on the back panel ANLG 1 / 2 IN BNC set the retardance: 0 - 5 V DC -> 0 - 25 V RMS drive.
        Needs "v1" or "v2" mode (the mode only chooses which stored voltage the analog signal replaces); the output
        must be on, see set_output(). Analog in and fast switching exclude each other on the controller.
        """
        self._check_channel(channel)
        if mode not in ("v1", "v2"):
            raise ValueError('analog control works in "v1" or "v2" mode only')
        self.set_fast_switching(channel, False)
        self.set_output_mode(channel, mode)
        self.set_analog_in(channel, True)
        if not self.get_analog_in(channel):
            raise RuntimeError(f"LCC25D channel {channel + 1} did not accept analog control "
                               f"(check the output mode and that trigger in is off)")
        self.logg.info(f"LCC25D channel {channel + 1} follows ANALOG IN (0 - {ANALOG_IN_MAX_V:g} V -> "
                       f"0 - {MAX_VOLTAGE:g} V RMS)")

    def disable_analog_control(self, channel):
        """Back to the stored Voltage 1 / 2."""
        self.set_analog_in(channel, False)

    def set_fast_switching(self, channel, on, v3_time_ms=None, v4_time_ms=None):
        """Fast switching mode (Voltage 3 / 4); the times (1 .. 50 ms) are the dwell on Voltage 3 and Voltage 4."""
        self._check_channel(channel)
        for index, t in ((0, v3_time_ms), (1, v4_time_ms)):
            if t is not None:
                self._check_range("voltage time", t, 1, 50)
                self._lcc.set_channel_voltage_time(channel, index, int(t))
        self._lcc.set_channel_fast_switching_state(channel, 1 if on else 0)

    # ------------------------------------------------------------------ sequence
    def set_sequence(self, channel, voltages, times_ms, cycles=1):
        """
        Load a voltage sequence (up to 512 steps): step i holds voltages[i] volts for times_ms[i] milliseconds
        (1 .. 5000), and the whole list is repeated `cycles` times when started with start_sequence().
        """
        self._check_channel(channel)
        n = len(voltages)
        if n != len(times_ms) or not 1 <= n <= MAX_SEQUENCE_STEPS:
            raise ValueError(f"need 1 .. {MAX_SEQUENCE_STEPS} steps and one time per voltage")
        for v in voltages:
            self._check_range("voltage", v, 0.0, MAX_VOLTAGE)
        for t in times_ms:
            self._check_range("step time", t, MIN_SEQUENCE_MS, MAX_SEQUENCE_MS)
        self._lcc.set_channel_output_mode(channel, OUTPUT_MODES["sequence"])
        self._lcc.set_channel_sequence_cycle(channel, int(cycles))
        self._lcc.set_channel_sequence(channel, list(range(1, n + 1)), [float(v) for v in voltages],
                                       [int(t) for t in times_ms], n)
        self.logg.info(f"LCC25D channel {channel + 1} sequence of {n} steps x {cycles} cycles loaded")

    def get_sequence(self, channel, max_steps=MAX_SEQUENCE_STEPS):
        """(voltages, times_ms) of the stored sequence."""
        self._check_channel(channel)
        index, voltage, times, length = [0] * max_steps, [0.0] * max_steps, [0] * max_steps, [0]
        self._lcc.get_channel_sequence(channel, index, voltage, times, max_steps, length)
        return voltage, times

    def start_sequence(self, channel):
        self.set_output(channel, True)

    def stop_sequence(self, channel):
        self.set_output(channel, False)

    def sequence_cycles_left(self, channel):
        """Cycles still to run; 0 once the sequence has finished."""
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_sequence_cycle(channel, out)
        return out[0]

    def wait_sequence(self, channel, timeout=60.0, poll=0.1):
        """Block until the sequence has run its cycles; False if timeout seconds pass first."""
        deadline = time.monotonic() + timeout
        while self.sequence_cycles_left(channel) > 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(poll)
        return True

    # ------------------------------------------------------------------ temperature
    def set_target_temperature(self, channel, celsius):
        """Liquid crystal temperature set point, 10 .. 50 C (the heater has to be on, see set_heating())."""
        self._check_channel(channel)
        self._check_range("temperature", celsius, MIN_TEMPERATURE, MAX_TEMPERATURE)
        self._lcc.set_channel_target_temperature(channel, float(celsius))

    def get_target_temperature(self, channel):
        self._check_channel(channel)
        out = [0.0]
        self._lcc.get_channel_target_temperature(channel, out)
        return out[0]

    def get_temperature(self, channel):
        self._check_channel(channel)
        out = [0.0]
        self._lcc.get_channel_actual_temperature(channel, out)
        return out[0]

    def set_heating(self, channel, on):
        """Only with a temperature-stabilized LC (TH10K sensor) connected; with none the controller shows Error."""
        self._check_channel(channel)
        self._lcc.set_channel_heating_state(channel, 1 if on else 0)

    def get_heating(self, channel):
        self._check_channel(channel)
        out = [0]
        self._lcc.get_channel_heating_state(channel, out)
        return bool(out[0])

    # ------------------------------------------------------------------ misc
    def restore_defaults(self):
        self._lcc.restore_default_parameters()
        self.logg.info("LCC25D default parameters restored")


if __name__ == "__main__":
    # Demo, like LCC25D_COMMAND_LIB_EXAMPLE.py: python -m minimiao.devices.lcc25d_retarder
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with LCC25DRetarder(logg=logging.getLogger("lcc25d")) as lcc:
        ch = 0
        lcc.set_voltage(ch, 0, 5.0)
        lcc.set_voltage(ch, 1, 12.0)
        print("Voltage 1/2:", lcc.get_voltage(ch, 0), lcc.get_voltage(ch, 1))
        lcc.set_modulation_frequency(ch, 10)
        lcc.set_output_mode(ch, "v1")
        lcc.set_output(ch, True)
        time.sleep(1)
        lcc.select_state(ch, 2)
        time.sleep(1)
        lcc.set_sequence(ch, [5.0, 6.0, 7.0, 8.0], [10, 10, 10, 10], cycles=10)
        lcc.start_sequence(ch)
        print("sequence finished:", lcc.wait_sequence(ch))
        print("sequence:", lcc.get_sequence(ch))
        lcc.set_output(ch, False)
        print("state:", lcc.get_channel_state(ch), "temperature:", lcc.get_temperature(ch))
