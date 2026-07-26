# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np

from minimiao import logger


class TriggerSequence:

    def __init__(self, sample_rate=2.0e5, logg=None):
        self.logg = logg or logger.setup_logging()
        # daq
        self.sample_rate = sample_rate  # Hz
        # digital triggers
        self.digital_starts = [0.00000, 0.0003, 0.0003, 0.00025, 0.00025, 0.0009]
        self.digital_ends = [0.00020, 0.0008, 0.0008, 0.003, 0.003, 0.001]
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
        # piezo scanner
        self.piezo_conv_factors = [10., 10., 10.]
        self.piezo_steps = [0.032, 0.032, 0.16]
        self.piezo_ranges = [0.16, 0.16, 0.8]
        self.piezo_positions = [30., 30., 30.]
        self.piezo_return_time = 0.06
        self.return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
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
        self.piezo_scan_dlt = 0.25  # s
        # TTL
        self.trigger_pulse_width = 10.0e-5  # s
        self.trigger_pulse_samples = int(np.ceil(self.trigger_pulse_width * self.sample_rate))
        self.cycle_time = 50.0e-3  # s
        self.cycle_samples = int(np.ceil(self.cycle_time * self.sample_rate))
        # SLM
        self.slm_delay_time = 0.00001  # s
        self.slm_delay_samples = round(self.slm_delay_time * self.sample_rate)
        self.slm_start_time = 270.187e-6  # s
        self.slm_start_samples = round(self.slm_start_time * self.sample_rate)
        self.slm_total_time = 2 * 20310.72e-6  # s
        self.slm_total_samples = round(self.slm_total_time * self.sample_rate)
        self.slm_end_time = 20270.187e-6 + self.slm_total_time / 2  # s
        self.slm_end_samples = round(self.slm_end_time * self.sample_rate)
        self.slm_on_time = 2 * 20000.0e-6 + 310.72e-6  # s
        self.slm_on_samples = round(self.slm_on_time * self.sample_rate)
        # camera
        self.initial_time = 0.000  # s
        self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        self.exposure_time = 0.04  # s
        self.exposure_samples = int(np.ceil(self.exposure_time * self.sample_rate))
        self.standby_time = 0.03893  # s
        self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        self.frame_time = 0.05  # s
        self.frame_samples = int(np.ceil(self.frame_time * self.sample_rate))
        # motor
        self.motor_jog_pulse = 0.001
        self.motor_jog_samples = int(np.ceil(self.motor_jog_pulse * self.sample_rate))
        self.motor_rot_time = 0.05
        self.motor_rot_samples = int(np.ceil(self.motor_rot_time * self.sample_rate))

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def update_sampling_rate(self, sample_rate=None):
        if sample_rate is not None:
            self.sample_rate = sample_rate  # Hz

    def update_piezo_scan_parameters(self, piezo_ranges=None, piezo_steps=None, piezo_positions=None,
                                     piezo_return_time=None, piezo_scan_dlt=None):
        original_values = {"piezo_ranges": self.piezo_ranges, "piezo_steps": self.piezo_steps,
                           "piezo_positions": self.piezo_positions,
                           "piezo_return_time": self.piezo_return_time, "piezo_scan_dlt": self.piezo_scan_dlt}
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
                self.return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
            if piezo_scan_dlt is not None:
                self.piezo_scan_dlt = piezo_scan_dlt
            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.piezo_scan_positions = [start + step * np.arange(ns) for start, step, ns in
                                         zip(self.piezo_starts, self.piezo_steps, self.piezo_scan_pos)]
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

    def update_camera_parameters(self, initial_time=None, exposure_time=None, standby_time=None, frame_rate=None):
        if initial_time is not None:
            self.initial_time = initial_time
            self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        if exposure_time is not None:
            self.exposure_time = exposure_time
            self.exposure_samples = int(np.ceil(self.exposure_time * self.sample_rate))
        if standby_time is not None:
            self.standby_time = standby_time
            self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        if frame_rate is not None:
            self.frame_time = 1 / frame_rate
            self.frame_samples = int(np.ceil(self.frame_time * self.sample_rate))

    def update_slm_parameters(self, total_time=None, start_time=None, on_time=None, end_time=None, delay_time=None):
        if total_time is not None:
            self.slm_total_time = total_time
            self.slm_total_samples = round(self.slm_total_time * self.sample_rate)
        if start_time is not None:
            self.slm_start_time = start_time
            self.slm_start_samples = round(self.slm_start_time * self.sample_rate)
        if on_time is not None:
            self.slm_on_time = on_time
            self.slm_on_samples = round(self.slm_on_time * self.sample_rate)
        if end_time is not None:
            self.slm_end_time = end_time
            self.slm_end_samples = round(self.slm_end_time * self.sample_rate)
        if delay_time is not None:
            self.slm_delay_time = delay_time
            self.slm_delay_samples = round(self.slm_delay_time * self.sample_rate)

    def generate_digital_triggers(self, lasers, camera):
        digital_channels = [2, 3]
        self.cycle_samples = int(max(self.slm_total_samples, self.frame_samples) * 1.5)
        self.cycle_time = self.cycle_samples / self.sample_rate
        digital_triggers = np.zeros((2, self.cycle_samples), dtype=np.uint8)
        digital_triggers[0, :self.trigger_pulse_samples] = 1
        digital_triggers[1, self.slm_start_samples:self.slm_start_samples + self.trigger_pulse_samples] = 1
        return digital_triggers, digital_channels

    def generate_sim_triggers(self, nph=6):
        digital_channels = [2, 3, 4, 5, 6]
        self.frame_samples = int(np.ceil(0.08 * self.sample_rate))
        self.cycle_samples = max(self.slm_total_samples, self.frame_samples) + 2
        self.cycle_time = self.cycle_samples / self.sample_rate
        digital_triggers = np.zeros((2, self.cycle_samples), dtype=np.uint8)
        digital_triggers[0, :self.trigger_pulse_samples] = 1
        digital_triggers[1, self.slm_start_samples:self.slm_start_samples + self.trigger_pulse_samples] = 1
        digital_triggers = np.tile(digital_triggers, (1, nph))
        digital_triggers = np.concatenate((digital_triggers, np.zeros((2, self.motor_rot_samples), dtype=np.uint8)), axis=1)
        digital_triggers = np.tile(digital_triggers, (1, 2))
        motor_stay = np.ones((3, self.cycle_samples * nph), dtype=np.uint8)
        motor_stay[0, -4 * self.motor_jog_samples:] = 0
        motor_fwd = np.ones((3, self.motor_rot_samples), dtype=np.uint8)
        motor_fwd[0, :5 * self.motor_jog_samples] = 0
        motor_fwd[2, :self.motor_jog_samples] = 0
        motor_bwd = np.ones((3, self.motor_rot_samples), dtype=np.uint8)
        motor_bwd[0, :5 * self.motor_jog_samples] = 0
        motor_bwd[1, :self.motor_jog_samples] = 0
        digit_triggers = np.concatenate((motor_stay, motor_fwd), axis=1)
        digit_triggers = np.concatenate((digit_triggers, motor_stay), axis=1)
        digit_triggers = np.concatenate((digit_triggers, motor_bwd), axis=1)
        return np.concatenate((digital_triggers, digit_triggers), axis=0), digital_channels

    def generate_piezo_scan(self, num, lasers, camera):
        digital_channels = [2, 3]
        self.cycle_samples = max(self.slm_total_samples, self.frame_samples) + 2
        self.cycle_time = self.cycle_samples / self.sample_rate
        stage_samples = (num - 1) * self.cycle_samples + max(self.slm_end_samples + self.return_samples, self.frame_samples)
        digital_triggers = np.zeros((2, stage_samples), dtype=np.uint8)
        for i in range(num):
            start = i * self.cycle_samples
            digital_triggers[0, start:start + self.trigger_pulse_samples] = 1
            digital_triggers[1, start + self.slm_start_samples:start + self.slm_start_samples + self.trigger_pulse_samples] = 1
        pos = 1
        pz_chs = []
        for i in range(3):
            if self.piezo_scan_pos[i] > 0:
                pos *= self.piezo_scan_pos[i]
                pz_chs.append(i)
        if len(pz_chs) == 0:
            raise Exception("Error: zero piezo scan step")
        piezo_sequences = [np.empty((0,)) for _ in range(len(pz_chs))]
        for n, pch in enumerate(pz_chs):
            piezo_sequences[n] = np.repeat(self.piezo_scan_positions[pch], digital_triggers.shape[1])
            piezo_sequences[n] = shift_array(piezo_sequences[n], self.return_samples,
                                             fill=piezo_sequences[n][0], direction="backward")
            for i in range(n):
                piezo_sequences[i] = np.tile(piezo_sequences[i], self.piezo_scan_pos[pch])
            digital_triggers = np.tile(digital_triggers, self.piezo_scan_pos[pch])
        return digital_triggers, convert_list(piezo_sequences), digital_channels, pz_chs, pos


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
