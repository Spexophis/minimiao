# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np

from minimiao import logger


class TriggerSequence:

    def __init__(self, dev, sample_rate=80e3, logg=None):
        self.galvo = dev
        self.logg = logg or logger.setup_logging()
        # daq
        self.sample_rate = sample_rate  # Hz
        # digital triggers
        self.digital_starts = [0.00000, 0.00025, 0.0008, 0.0008, 0.0008, 0.0008]
        self.digital_ends = [0.00020, 0.00075, 0.001, 0.001, 0.001, 0.001]
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
        # galvo scanner
        self.galvo_ramp_time = 300  # ~300 us
        self.galvo_ramp_time_samples = round(self.galvo_ramp_time * 1e-6 * self.sample_rate)
        self.galvo_step_response = 400  # ~325 us
        self.galvo_step_response_samples = round(self.galvo_step_response * 1e-6 * self.sample_rate)
        self.galvo_return_time = 300  # ~500 us
        self.galvo_return_samples = round(self.galvo_return_time * 1e-6 * self.sample_rate)
        self.galvo_origins = [0.0, -0.1]  # V
        self.galvo_ranges = [0.1, 0.2]  # V
        self.galvo_offsets = [-0.0, 0.0]  # V
        self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
        self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
        self.dot_steps = [0.02, 0.04]  # volts
        self.dot_pos = [np.arange(dot_start, galvo_stop, dot_step) for (dot_start, galvo_stop, dot_step) in
                        zip(self.galvo_starts, self.galvo_stops, self.dot_steps)]
        self.galvo_scan_pos = [dps.size for dps in self.dot_pos]
        # piezo scanner
        self.piezo_conv_factors = [10.]
        self.piezo_steps = [0.08]
        self.piezo_ranges = [2.0]
        self.piezo_positions = [50.]
        self.piezo_return_time = 0.04  # ~50 ms
        self.piezo_return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                            zip(self.piezo_steps, self.piezo_conv_factors)]
        self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                             zip(self.piezo_ranges, self.piezo_conv_factors)]
        self.piezo_positions = [position / conv_factor for position, conv_factor in
                                zip(self.piezo_positions, self.piezo_conv_factors)]
        self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
        self.piezo_scan_pos = [int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                               zip(self.piezo_ranges, self.piezo_steps)]
        self.piezo_scan_positions = [start + step * np.arange(ns) for start, step, ns in
                                     zip(self.piezo_starts, self.piezo_steps, self.piezo_scan_pos)]
        # GUI & Thread
        self.refresh_time = 0.006  # s
        self.refresh_time_samples = round(self.refresh_time * self.sample_rate)
        # camera
        self.initial_time = 0.0016  # s
        self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        self.standby_time = 0.0032  # s
        self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        self.exposure_time = 0.0064  # s
        self.exposure_samples = int(np.ceil(self.exposure_time * self.sample_rate))
        self.trigger_pulse_width = 1e-4  # s
        self.trigger_pulse_samples = int(np.ceil(self.trigger_pulse_width * self.sample_rate))

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def update_sampling_rate(self, sample_rate=None):
        if sample_rate is not None:
            self.sample_rate = sample_rate  # Hz

    def update_galvo_scan_parameters(self, origins=None, ranges=None, foci=None, offsets=None, returns=None):
        original_values = {"galvo_origins": self.galvo_origins, "galvo_ranges": self.galvo_ranges,
                           "galvo_starts": self.galvo_starts, "galvo_stops": self.galvo_stops,
                           "galvo_offset": self.galvo_offsets, "dot_steps": self.dot_steps, "dot_pos": self.dot_pos,
                           "galvo_return_time": self.galvo_return_time, "galvo_step_response": self.galvo_step_response}
        try:
            if origins is not None:
                self.galvo_origins = origins
            if ranges is not None:
                self.galvo_ranges = ranges
            if foci is not None:
                self.dot_steps = foci
            if offsets is not None:
                self.galvo_offsets = offsets
            if returns is not None:
                self.galvo_ramp_time, self.galvo_return_time, self.galvo_step_response = returns
            self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.dot_pos = [np.arange(dot_start, galvo_stop, dot_step) for (dot_start, galvo_stop, dot_step) in
                            zip(self.galvo_starts, self.galvo_stops, self.dot_steps)]
            self.galvo_scan_pos = [dps.size for dps in self.dot_pos]
            self.galvo_ramp_time_samples = round(self.galvo_ramp_time * 1e-6 * self.sample_rate)
            self.galvo_step_response_samples = round(self.galvo_step_response * 1e-6 * self.sample_rate)
            self.galvo_return_samples = round(self.galvo_return_time * 1e-6 * self.sample_rate)
            self.galvo.set_params(sample_rate_hz=self.sample_rate,
                                  start_s=self.galvo_starts, pixel_s=self.dot_steps, pixels_lines=self.galvo_scan_pos,
                                  t_ramp_us=self.galvo_ramp_time, t_flyback_us=self.galvo_return_time, t_settle_us=self.galvo_step_response)
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Galvo scanning parameters reverted to original values.")
            return

    def update_piezo_scan_parameters(self, piezo_ranges=None, piezo_steps=None, piezo_positions=None,
                                     piezo_return_time=None):
        original_values = {"piezo_ranges": self.piezo_ranges, "piezo_steps": self.piezo_steps,
                           "piezo_positions": self.piezo_positions, "piezo_return_time": self.piezo_return_time}
        try:
            if piezo_ranges is not None:
                self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                                     zip(piezo_ranges, self.piezo_conv_factors)]
            if piezo_steps is not None:
                self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                                    zip(piezo_steps, self.piezo_conv_factors)]
            if piezo_positions is not None:
                self.piezo_positions = [position / conv_factor for position, conv_factor in
                                        zip(piezo_positions, self.piezo_conv_factors)]
            if piezo_return_time is not None:
                self.piezo_return_time = piezo_return_time

            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.piezo_scan_positions = [start + step * np.arange(ns) for start, step, ns in
                                         zip(self.piezo_starts, self.piezo_steps, self.piezo_scan_pos)]
            self.piezo_return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Piezo scanning parameters reverted to original values.")
            return

    def update_digital_parameters(self, digital_starts=None, digital_ends=None):
        if digital_starts is not None:
            self.digital_starts = digital_starts
        if digital_ends is not None:
            self.digital_ends = digital_ends
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]

    def generate_digital_triggers(self, lasers, detectors):
        digital_channels = lasers.copy()
        if len(detectors) == 0:
            detect_ind = []
            gate_ind = [5]
        else:
            detect_ind = [detector + 3 for detector in detectors]
            digital_channels.extend(detect_ind)
        cycle_samples = max([self.digital_ends[i] for i in digital_channels])
        laser_triggers = np.zeros((len(lasers), cycle_samples), dtype=np.uint8)
        for ln, ch in enumerate(lasers):
            laser_triggers[ln, self.digital_starts[ch]:self.digital_ends[ch]] = 1
        pixel_dwell_samples = self.digital_ends[3] - self.digital_starts[3]
        if len(detect_ind) > 0:
            detector_triggers = np.zeros((len(detect_ind), cycle_samples), dtype=np.uint8)
            detector_trigger = np.sum(laser_triggers, axis=0)
            detector_triggers[:] = detector_trigger
            digital_triggers = np.vstack((laser_triggers, detector_triggers))
            detector_gates = np.zeros((len(detect_ind), cycle_samples), dtype=np.uint8)
            for ln, ch in enumerate(detect_ind):
                detector_gates[ln, self.digital_starts[ch]:self.digital_ends[ch]] = 1
        else:
            digital_triggers = laser_triggers
            detector_gates = np.zeros((len(gate_ind), cycle_samples), dtype=np.uint8)
            for ln, ch in enumerate(gate_ind):
                detector_gates[ln, self.digital_starts[ch]:self.digital_ends[ch]] = 1
        return digital_triggers, digital_channels, detector_gates, pixel_dwell_samples

    def generate_digital_triggers_for_galvo_scan(self, lasers, detectors):
        digital_triggers, chs, gates, dws = self.generate_digital_triggers(lasers, detectors)
        cycle_samples = digital_triggers.shape[1] + self.galvo_step_response_samples
        compensate_sequence = np.zeros((digital_triggers.shape[0], self.galvo_step_response_samples))
        digital_triggers = np.concatenate((digital_triggers, compensate_sequence), axis=1)
        compensate_sequence = np.zeros((gates.shape[0], self.galvo_step_response_samples))
        gates = np.concatenate((gates, compensate_sequence), axis=1)
        return digital_triggers, chs, cycle_samples, gates, dws

    def generate_galvo_resolft_scan(self, lasers, detectors):
        digital_triggers, dig_chs, cycle_samples, gates, dwl = self.generate_digital_triggers_for_galvo_scan(lasers, detectors)
        pos = 1
        gv_chs = []
        for i, nps in enumerate(self.galvo_scan_pos):
            if nps > 0:
                pos *= nps
                gv_chs.append(i)
        if len(gv_chs) == 0:
            raise Exception("Error: zero piezo scan step")
        galvo_sequences = [np.empty((0,)) for _ in range(len(gv_chs))]
        step_t = np.arange(self.galvo_step_response_samples) * 1e6 / self.sample_rate
        step_curve, _, _ = self.galvo.get_trajectory(step_t, p0=0, v0=0, p1=self.dot_steps[0], v1=0)
        step_up = np.zeros(digital_triggers.shape[1])
        step_up[-self.galvo_step_response_samples:] = step_curve
        n, pch = 0, 0
        galvo_sequences[n] = np.repeat(self.dot_pos[pch], digital_triggers.shape[1]) + np.tile(step_up, self.galvo_scan_pos[0])
        offset_samples = max(self.galvo_return_samples - self.galvo_step_response_samples, 0)
        galvo_sequences[n] = np.pad(galvo_sequences[n], (0, offset_samples), mode="constant",
                                    constant_values=galvo_sequences[n][0])
        return_t = np.arange(self.galvo_return_samples) * 1e6 / self.sample_rate
        return_curve, _, _ = self.galvo.get_trajectory(return_t, p0=self.dot_pos[pch][-1], v0=0, p1=self.dot_pos[pch][0], v1=0)
        galvo_sequences[n][-self.galvo_return_samples:] = return_curve
        digital_triggers = np.tile(digital_triggers, self.galvo_scan_pos[pch])
        gates = np.tile(gates, self.galvo_scan_pos[pch])
        digital_triggers = np.pad(digital_triggers, ((0, 0), (0, offset_samples)), mode="constant", constant_values=0)
        gates = np.pad(gates, ((0, 0), (0, offset_samples)), mode="constant", constant_values=0)
        n, pch = 1, 1
        galvo_sequences[n] = np.repeat(self.dot_pos[pch], digital_triggers.shape[1])
        step_curve, _, _ = self.galvo.get_trajectory(step_t, p0=0, v0=0, p1=self.dot_steps[1], v1=0)
        step_up = np.zeros(digital_triggers.shape[1])
        step_up[-self.galvo_step_response_samples:] = step_curve
        galvo_offset_y = np.linspace(0, self.galvo_offsets[1], digital_triggers.shape[1])
        galvo_offset_y = np.tile(galvo_offset_y, self.galvo_scan_pos[1])
        galvo_sequences[n] += galvo_offset_y + np.tile(step_up, self.galvo_scan_pos[1])
        galvo_offset_x = np.linspace(0, self.galvo_offsets[0], self.galvo_scan_pos[1])
        galvo_offset_x = np.repeat(galvo_offset_x, galvo_sequences[0].shape[0])
        galvo_sequences[0] = np.tile(galvo_sequences[0], self.galvo_scan_pos[pch])
        galvo_sequences[0] += galvo_offset_x
        digital_triggers = np.tile(digital_triggers, self.galvo_scan_pos[pch])
        gates = np.tile(gates, self.galvo_scan_pos[pch])
        return digital_triggers, convert_list(galvo_sequences), dig_chs, gv_chs, pos, gates, dwl

    def generate_piezo_resolft_scan(self, lasers, detectors):
        digital_triggers, galvo_sequences, dig_chs, gv_chs, pos, gates, dwl = self.generate_galvo_resolft_scan(lasers, detectors)

        return digital_triggers, convert_list(galvo_sequences), dig_chs, gv_chs, pos, gates, dwl

    def generate_galvo_resolft_static(self, lasers, detectors):
        digital_triggers, dig_chs, cycle_samples, gates, dwl = self.generate_digital_triggers_for_galvo_scan(lasers, detectors)
        pos = 16
        gv_chs = [0, 1]
        galvo_sequences = np.ones((2, digital_triggers.shape[1]))
        galvo_sequences[0] *= self.galvo_origins[0]
        galvo_sequences[1] *= self.galvo_origins[1]
        galvo_sequences = np.tile(galvo_sequences, pos)
        digital_triggers = np.tile(digital_triggers, pos)
        return digital_triggers, galvo_sequences, dig_chs, gv_chs, pos, gates, dwl

    def generate_galvo_point_scan(self, lasers, detectors):
        pos = 1
        gv_chs = []
        for i, nps in enumerate(self.galvo_scan_pos):
            if nps > 0:
                pos *= nps
                gv_chs.append(i)
        if len(gv_chs) == 0:
            raise Exception("Error: zero piezo scan step")
        galvo_sequences = [np.empty((0,)) for _ in range(len(gv_chs))]
        galvo_sequences[0], galvo_sequences[1], gate_ttl = self.galvo.build_scan()
        detect_ind = [detector + 3 for detector in detectors]
        digital_channels = lasers.copy()
        digital_channels.extend(detect_ind)
        digital_triggers = [gate_ttl for _ in range(len(digital_channels))]
        gates = [gate_ttl for _ in range(len(detect_ind))]
        return convert_list(digital_triggers), convert_list(galvo_sequences), digital_channels, gv_chs, pos, gates

    def generate_galvo_axial_scan(self, lasers, detectors):
        digital_triggers, galvo_sequences, digital_channels, gv_chs, pos, gates = self.generate_galvo_point_scan(lasers, detectors)
        return convert_list(digital_triggers), convert_list(galvo_sequences), digital_channels, gv_chs, pos, gates


def convert_list(arrays):
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.array(arrays)


def smooth_ramp(start, end, samples, curve_half=0.02):
    n = int(curve_half * samples)
    x = np.linspace(0, np.pi / 2, n, endpoint=True)
    signal_first_half = np.sin(x) * (end - start) / np.sin(np.pi / 2) + start
    signal_second_half = np.full(samples - n, end)
    return np.concatenate((signal_first_half, signal_second_half), dtype=np.float16)


def shift_array(arr, shift_length, fill=None, direction='backward'):
    if len(arr) == 0 or shift_length == 0:
        return arr
    shifted_array = np.empty_like(arr)
    shift_length = abs(shift_length) % len(arr)
    if fill is not None:
        last_element = fill
    else:
        if direction == 'forward':
            last_element = arr[0]
        elif direction == 'backward':
            last_element = arr[-1]
    if direction == 'forward':
        if shift_length < len(arr):
            shifted_array[shift_length:] = arr[:-shift_length]
        shifted_array[:shift_length] = last_element
    elif direction == 'backward':
        if shift_length < len(arr):
            shifted_array[:-shift_length] = arr[shift_length:]
        shifted_array[-shift_length:] = last_element
    return shifted_array


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0
