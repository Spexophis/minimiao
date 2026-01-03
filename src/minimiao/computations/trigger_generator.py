# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np
import os

class TriggerSequence:

    def __init__(self, sample_rate=2.5e5, logg=None):
        self.logg = logg or self.setup_logging()
        # daq
        self.sample_rate = sample_rate  # Hz
        # digital triggers
        self.digital_starts = [0.00000, 0.0003, 0.0003, 0.00025, 0.00025, 0.0009]
        self.digital_ends = [0.00025, 0.0008, 0.0008, 0.003, 0.003, 0.001]
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
        # piezo scanner
        self.piezo_conv_factors = [10., 10., 10.]
        self.piezo_steps = [0.032, 0.032, 0.16]
        self.piezo_ranges = [2.0, 2.0, 0.0]
        self.piezo_positions = [50., 50., 50.]
        self.piezo_return_time = 0.05
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
        # GUI & Thread
        self.refresh_time = 0.006  # s
        self.refresh_time_samples = int(np.ceil(self.refresh_time * self.sample_rate))
        # SLM
        self.slm_delay_time = 0.00001  # s
        self.slm_delay_samples = round(self.slm_delay_time * self.sample_rate)
        self.slm_start_time = 270.187e-6  # s
        self.slm_start_samples = round(self.slm_start_time * self.sample_rate)
        self.switch_off_time = 720.96e-6 - self.slm_start_time  # s
        self.switch_off_samples = round(self.switch_off_time * self.sample_rate)
        self.readout_time = 520.853e-6 - self.slm_start_time  # s
        self.readout_samples = round(self.readout_time * self.sample_rate)
        self.total_switch_on_time = 776.64e-6  # s
        self.total_switch_on_samples = round(self.total_switch_on_time * self.sample_rate)
        self.total_readout_time = 576.533e-6  # s
        self.total_readout_samples = round(self.total_readout_time * self.sample_rate)
        # camera
        self.initial_time = 0.00159  # s
        self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        self.standby_time = 0.03893  # s
        self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        self.exposure_time = 0.005  # s
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

    def update_camera_parameters(self, initial_time=None, standby_time=None):
        if initial_time is not None:
            self.initial_time = initial_time
            self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        if standby_time is not None:
            self.standby_time = standby_time
            self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))

    def generate_slm_triggers(self, slm_seq="5ms_dark_pair"):
        if "200us" in slm_seq:
            # "200us_lit_balanced"
            samps_total = round(576.533e-6 * self.sample_rate)
            expo_on = 199.893e-6
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(520.853e-6 * self.sample_rate)
            act_seq = np.zeros(samps_total, dtype=np.uint8)
            act_seq[:self.trigger_pulse_samples] = 1
            cam_seq = np.zeros(samps_total, dtype=np.uint8)
            cam_seq[samps_start:samps_end + self.slm_delay_samples] = 1
        if "400us" in slm_seq:
            # "400us_lit_balanced"
            samps_total = round(776.64e-6 * self.sample_rate)
            expo_on = 400e-6
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(720.96e-6 * self.sample_rate)
            act_seq = np.zeros(samps_total, dtype=np.uint8)
            act_seq[:self.trigger_pulse_samples] = 1
            cam_seq = np.zeros(samps_total, dtype=np.uint8)
            cam_seq[samps_start:samps_end + self.slm_delay_samples] = 1
        elif "600us" in slm_seq:
            # "600us_lit_balanced"
            samps_total = round(976.747e-6 * self.sample_rate)
            expo_on = 600.107e-6
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(921.067e-6 * self.sample_rate)
            act_seq = np.zeros(samps_total, dtype=np.uint8)
            act_seq[:self.trigger_pulse_samples] = 1
            cam_seq = np.zeros(samps_total, dtype=np.uint8)
            cam_seq[samps_start:samps_end + self.slm_delay_samples] = 1
        elif "5ms" in slm_seq:
            # "5ms_dark_pair"
            samps_total = round(5.31072e-3 * self.sample_rate)
            expo_on = 5.0e-3
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(5.270187e-3 * self.sample_rate)
            act_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            act_seq[:self.trigger_pulse_samples] = 1
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        elif "10ms" in slm_seq:
            # "10ms_dark_pair"
            samps_total = round(10.31072e-3 * self.sample_rate)
            expo_on = 10.0e-3
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(10.270187e-3 * self.sample_rate)
            act_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            act_seq[:self.trigger_pulse_samples] = 1
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        elif "20ms" in slm_seq:
            # "20ms_dark_pair"
            samps_total = round(20.31072e-3 * self.sample_rate)
            expo_on = 20.0e-3
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(20.270187e-3 * self.sample_rate)
            act_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            act_seq[:self.trigger_pulse_samples] = 1
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        else:
            self.logg.error("SLM sequence error.")
            raise ValueError("SLM sequence is wrong.")
        return act_seq, cam_seq, expo_on, samps_on

    def generate_sim_triggers(self, lasers, camera, slm_seq, dim):
        cam_ind = camera + 3
        digital_channels = lasers.copy()
        digital_channels.append(cam_ind)
        interval_samples = self.initial_samples
        act_seq, cam_seq, self.exposure_time, self.exposure_samples = self.generate_slm_triggers(slm_seq)
        dark_samples = int(act_seq.shape[0] / 2)
        offset_samples = max(self.standby_samples - dark_samples, 0) + int(1e-3 * self.sample_rate)
        if len(digital_channels) == 2:
            cycle_samples = interval_samples + act_seq.shape[0] + offset_samples
            digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
            digital_trigger[0][interval_samples:interval_samples + act_seq.shape[0]] = act_seq
            digital_trigger[1][interval_samples:interval_samples + cam_seq.shape[0]] = cam_seq
        elif len(digital_channels) == 3:
            expo_samples = int(self.digital_ends[lasers[0]] - self.digital_starts[lasers[0]])
            interp_samples = max(interval_samples, expo_samples)
            cycle_samples = interp_samples + act_seq.shape[0] + offset_samples
            digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
            digital_trigger[0][:expo_samples] = 1
            digital_trigger[1][interp_samples:interp_samples + act_seq.shape[0]] = act_seq
            digital_trigger[2][interp_samples:interp_samples + cam_seq.shape[0]] = cam_seq
        else:
            self.logg.error("Digital channels error.")
            raise ValueError("Digital channels number is wrong.")
        if dim == 2:
            digital_trigger = np.tile(digital_trigger, (1, 3))
        elif dim == 3:
            digital_trigger = np.tile(digital_trigger, (1, 5))
        else:
            self.logg.error("SIM dimension error.")
            raise ValueError("SIM dimension number is wrong.")
        return digital_trigger, digital_channels

    def generate_digital_triggers(self, lasers, camera, slm_seq):
        cam_ind = camera + 3
        digital_channels = lasers.copy()
        digital_channels.append(cam_ind)
        interval_samples = self.initial_samples
        if slm_seq == "None":
            if interval_samples > self.digital_starts[cam_ind]:
                offset_samples = interval_samples - self.digital_starts[cam_ind]
                self.digital_starts = [(_start + offset_samples) for _start in self.digital_starts]
                self.digital_ends = [(_end + offset_samples) for _end in self.digital_ends]
            cycle_samples = max(self.digital_ends[cam_ind] + self.standby_samples + 2,
                                max([self.digital_ends[i] for i in digital_channels]))
            digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
            self.exposure_samples = self.digital_ends[cam_ind] - self.digital_starts[cam_ind]
            self.exposure_time = self.exposure_samples / self.sample_rate
            for ln, ch in enumerate(digital_channels):
                digital_trigger[ln, self.digital_starts[ch]:self.digital_ends[ch]] = 1
        else:
            act_seq, cam_seq, self.exposure_time, self.exposure_samples = self.generate_slm_triggers(slm_seq)
            dark_samples = int(act_seq.shape[0] / 2)
            offset_samples = max(self.standby_samples - dark_samples, 0)
            if len(digital_channels) == 2:
                cycle_samples = interval_samples + act_seq.shape[0] + offset_samples
                digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
                digital_trigger[0][interval_samples:interval_samples + act_seq.shape[0]] = act_seq
                digital_trigger[1][interval_samples:interval_samples + cam_seq.shape[0]] = cam_seq
            elif len(digital_channels) == 3:
                expo_samples = int(self.digital_ends[lasers[0]] - self.digital_starts[lasers[0]])
                interp_samples = max(interval_samples, expo_samples)
                cycle_samples = interp_samples + act_seq.shape[0] + offset_samples
                digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
                digital_trigger[0][:expo_samples] = 1
                digital_trigger[1][interp_samples:interp_samples + act_seq.shape[0]] = act_seq
                digital_trigger[2][interp_samples:interp_samples + cam_seq.shape[0]] = cam_seq
            else:
                self.logg.error("Digital channels error.")
                raise ValueError("Digital channels number is wrong.")
        return digital_trigger, digital_channels

    def generate_sim_triggers_for_3d(self, lasers, camera, slm_seq):
        digital_triggers, chs = self.generate_sim_triggers(lasers, camera, slm_seq, 3)
        if self.standby_samples > self.return_samples:
            cycle_samples = digital_triggers.shape[1]
        else:
            compensate_samples = self.return_samples - self.standby_samples
            cycle_samples = digital_triggers.shape[1] + compensate_samples
            compensate_sequence = np.zeros((digital_triggers.shape[0], compensate_samples))
            digital_triggers = np.concatenate((digital_triggers, compensate_sequence), axis=1)
        return digital_triggers, cycle_samples, chs

    def generate_digital_triggers_for_scan(self, lasers, camera, slm_seq):
        digital_triggers, chs = self.generate_digital_triggers(lasers, camera, slm_seq)
        if self.standby_samples > self.return_samples:
            cycle_samples = digital_triggers.shape[1]
        else:
            compensate_samples = self.return_samples - self.standby_samples
            cycle_samples = digital_triggers.shape[1] + compensate_samples
            compensate_sequence = np.zeros((digital_triggers.shape[0], compensate_samples))
            digital_triggers = np.concatenate((digital_triggers, compensate_sequence), axis=1)
        return digital_triggers, cycle_samples, chs

    def generate_sim_3d(self, lasers, camera, slm_seq):
        digital_triggers, cycle_samples, dig_chs = self.generate_sim_triggers_for_3d(lasers, camera, slm_seq)
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
            piezo_sequences[n] = shift_array(piezo_sequences[n], max(self.standby_samples, self.return_samples),
                                             fill=piezo_sequences[n][0], direction="backward")
            for i in range(n):
                piezo_sequences[i] = np.tile(piezo_sequences[i], self.piezo_scan_pos[pch])
            digital_triggers = np.tile(digital_triggers, self.piezo_scan_pos[pch])
        return digital_triggers, convert_list(piezo_sequences), dig_chs, pz_chs, pos

    def generate_piezo_scan(self, lasers, camera, slm_seq):
        digital_triggers, cycle_samples, dig_chs = self.generate_digital_triggers_for_scan(lasers, camera, slm_seq)
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
            piezo_sequences[n] = shift_array(piezo_sequences[n], max(self.standby_samples, self.return_samples),
                                             fill=piezo_sequences[n][0], direction="backward")
            for i in range(n):
                piezo_sequences[i] = np.tile(piezo_sequences[i], self.piezo_scan_pos[pch])
            digital_triggers = np.tile(digital_triggers, self.piezo_scan_pos[pch])
        return digital_triggers, convert_list(piezo_sequences), dig_chs, pz_chs, pos

    def generate_live_point_scan_2d(self, lasers, slm_seq):
        digital_channels = lasers.copy()
        digital_channels.append(5)
        digital_sequences = []
        if slm_seq == "None":
            line_samples = int(self.digital_ends[5] + self.sample_rate * 5e-5)
            n = 1 + int(self.refresh_time_samples / line_samples)
            if 0 in lasers:
                switch_on_sequence = np.zeros(line_samples, dtype=np.uint8)
                switch_on_sequence[self.digital_starts[0]:self.digital_ends[0]] = 1
                switch_on_sequence = np.tile(switch_on_sequence, n)
                digital_sequences.append(switch_on_sequence)
            if 2 in lasers:
                switch_off_sequence = np.zeros(line_samples, dtype=np.uint8)
                switch_off_sequence[self.digital_starts[2]:self.digital_ends[2]] = 1
                switch_off_sequence = np.tile(switch_off_sequence, n)
                digital_sequences.append(switch_off_sequence)
            if 5 in digital_channels:
                readout_sequence = np.zeros(line_samples, dtype=np.uint8)
                readout_sequence[self.digital_starts[5]:self.digital_ends[5]] = 1
                readout_sequence = np.tile(readout_sequence, n)
                digital_sequences.append(readout_sequence)
            pixel_dwell_sample = self.digital_ends[5] - self.digital_starts[5]
        else:
            if self.slm_start_samples > self.digital_ends[0]:
                offset_samples =  self.slm_start_samples - self.digital_ends[0]
                self.digital_starts = [(_start + offset_samples) for _start in self.digital_starts]
                self.digital_ends = [(_end + offset_samples) for _end in self.digital_ends]
            line_samples = int(self.digital_ends[0] + int(np.ceil(1.4e-3 * self.sample_rate)))
            if 0 in lasers:
                switch_on_sequence = np.zeros(line_samples, dtype=np.uint8)
                switch_on_sequence[self.digital_starts[0]:self.digital_ends[0]] = 1
                digital_sequences.append(switch_on_sequence)
            start_ = 0
            end_ = 0
            if 2 in lasers:
                switch_off_sequence = np.zeros(line_samples, dtype=np.uint8)
                start_ += max(self.digital_ends[0] - self.slm_start_samples - self.slm_delay_samples, 0)
                switch_off_sequence[start_:start_ + self.trigger_pulse_samples] = 1
                digital_sequences.append(switch_off_sequence)
            if 5 in digital_channels:
                readout_sequence = np.zeros(line_samples, dtype=np.uint8)
                start_ += self.total_switch_on_samples + self.slm_start_samples
                end_ += start_ + self.readout_samples
                readout_sequence[start_:end_ + self.slm_delay_samples] = 1
                digital_sequences.append(readout_sequence)
            pixel_dwell_sample = self.readout_samples + self.slm_delay_samples
        return np.asarray(digital_sequences), digital_channels, pixel_dwell_sample

    def generate_piezo_point_scan_2d(self, lasers):
        line_samples = int(self.piezo_scan_dlt * self.sample_rate)
        fast_x = np.linspace(self.piezo_positions[0] - self.piezo_ranges[0] / 2,
                             self.piezo_positions[0] + self.piezo_ranges[0] / 2,
                             line_samples)
        slow_y = np.linspace(self.piezo_positions[0] - self.piezo_ranges[0] / 2,
                             self.piezo_positions[0] + self.piezo_ranges[0] / 2,
                             self.piezo_scan_pos[1])
        fast_x = np.tile(fast_x, self.piezo_scan_pos[1])
        slow_y = np.repeat(slow_y, line_samples)
        dwell_samples = int(line_samples / self.piezo_scan_pos[0])
        digital_channels = lasers.copy()
        digital_channels.append(5)
        n = line_samples // dwell_samples
        digital_sequences = []
        if self.slm_start_samples > self.digital_ends[0]:
            offset_samples =  self.slm_start_samples - self.digital_ends[0]
            self.digital_starts = [(_start + offset_samples) for _start in self.digital_starts]
            self.digital_ends = [(_end + offset_samples) for _end in self.digital_ends]
        if 0 in lasers:
            switch_on_sequence = np.zeros(line_samples, dtype=np.uint8)
            switch_on_sequence[:n * dwell_samples].reshape(n, dwell_samples)[:, self.digital_starts[0]:self.digital_ends[0]] = 1
            digital_sequences.append(switch_on_sequence)
        start_ = 0
        end_ = 0
        if 2 in lasers:
            switch_off_sequence = np.zeros(line_samples, dtype=np.uint8)
            start_ += max(self.digital_ends[0] - self.slm_start_samples - self.slm_delay_samples, 0)
            switch_off_sequence[:n * dwell_samples].reshape(n, dwell_samples)[:, start_:start_ + self.trigger_pulse_samples] = 1
            digital_sequences.append(switch_off_sequence)
        if 5 in digital_channels:
            readout_sequence = np.zeros(line_samples, dtype=np.uint8)
            start_ += self.total_switch_on_samples + self.slm_start_samples
            end_ += start_ + self.readout_samples
            readout_sequence[:n * dwell_samples].reshape(n, dwell_samples)[:, start_:end_ + self.slm_delay_samples] = 1
            digital_sequences.append(readout_sequence)
        digital_sequences = np.tile(np.array(digital_sequences), (1, self.piezo_scan_pos[1]))
        pixel_dwell_sample = self.readout_samples + self.slm_delay_samples
        return np.vstack((fast_x, slow_y)), [0, 1], digital_sequences, digital_channels, pixel_dwell_sample

    # def generate_piezo_point_scan_2d(self, lasers, camera, span="1 um"):
    #     ramp_libs = {"1 um": self.generate_piezo_scan_ramp_1um(),
    #                  "2 um": self.generate_piezo_scan_ramp_2um()}
    #     fast_x, slow_y, fv, samples_x, line, offset = ramp_libs[span]
    #     cam_ind = camera + 2
    #     digital_channels = lasers.copy()
    #     initial_offset = self.initial_samples
    #     if initial_offset > self.digital_starts[cam_ind]:
    #         initial_offset -= self.digital_starts[cam_ind]
    #         self.digital_starts = [(_start + initial_offset) for _start in self.digital_starts]
    #         self.digital_ends = [(_end + initial_offset) for _end in self.digital_ends]
    #     self.exposure_samples = self.digital_ends[cam_ind] - self.digital_starts[cam_ind]
    #     self.exposure_time = self.exposure_samples / self.sample_rate
    #     interval_samples = int(self.sample_rate * self.piezo_steps[0] * 10 / fv)
    #     if interval_samples < self.standby_samples + self.exposure_samples:
    #         raise Exception("Error: step size is less than the sum of camera exposure and readout")
    #     cycle_samples = self.digital_ends[cam_ind] + interval_samples + 2
    #     digital_channels.append(cam_ind)
    #     digital_sequences = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
    #     for ln, chl in enumerate(digital_channels):
    #         digital_sequences[ln, self.digital_starts[chl]:self.digital_ends[chl]] = 1
    #     digital_sequences = np.tile(digital_sequences, 32)
    #     offset_samples = np.zeros((len(digital_channels), int((0.025 + offset) / (0.2 / samples_x))))
    #     digital_sequences = np.concatenate((offset_samples, digital_sequences), axis=1)
    #     offset_samples = np.zeros((len(digital_channels), line.shape[0] - digital_sequences.shape[1]))
    #     digital_sequences = np.concatenate((digital_sequences, offset_samples), axis=1)
    #     digital_sequences = np.tile(digital_sequences, 32)
    #     offset_samples = np.zeros((len(digital_channels), samples_x + 2000))
    #     digital_sequences = np.concatenate((offset_samples, digital_sequences), axis=1)
    #     return np.vstack((fast_x, slow_y)), digital_sequences, digital_channels, [0, 1], 32 * 32
    #
    # def generate_piezo_scan_ramp_1um(self):
    #     lt = 0.25
    #     lnm = 32
    #     fv = 2 / lt
    #     starts = [s - 0.1 for s in self.piezo_positions]
    #     starts[1] += 0.025
    #     ends = [s + 0.1 for s in self.piezo_positions]
    #     offset = 0.03
    #     samples_x = int(lt * self.sample_rate)
    #     d_0 = np.linspace(start=starts[0], stop=ends[0], num=samples_x, endpoint=True)
    #     d_1 = starts[0] * np.ones(2000) - offset
    #     fast_x = np.concatenate((d_0, d_1))
    #     d_2 = np.linspace(start=starts[0] - offset, stop=ends[0] - offset, num=samples_x,
    #                       endpoint=True)
    #     dx = d_2[1] - d_2[0]
    #     d_extension = np.linspace(d_2[-1] + dx, d_2[-1] + offset, num=int(offset / dx), endpoint=True)
    #     d_2 = np.concatenate((d_2, d_extension))
    #     d_2[:3500] = np.linspace(start=starts[0] - offset, stop=d_2[3500] + 0.025, num=3500, endpoint=True)
    #     d_2[3500:] += 0.0025
    #     line = np.concatenate((d_2, d_1))
    #     for i in range(lnm):
    #         fast_x = np.concatenate((fast_x, line))
    #     slow_y = np.arange(starts[1], ends[1], step=self.piezo_steps[1])
    #     slow_y = slow_y[:lnm]
    #     slow_y = np.repeat(slow_y, line.shape[0])
    #     slow_y = np.concatenate((np.ones(samples_x + 2000) * starts[1], slow_y))
    #     return fast_x, slow_y, fv, samples_x, line, offset
    #
    # def generate_piezo_scan_ramp_2um(self):
    #     lt = 0.25
    #     lnm = 32
    #     fv = 2 / lt
    #     starts = [s - 0.1 for s in self.piezo_positions]
    #     starts[1] += 0.025
    #     ends = [s + 0.1 for s in self.piezo_positions]
    #     offset = 0.03
    #     samples_x = int(lt * self.sample_rate)
    #     d_0 = np.linspace(start=starts[0], stop=ends[0], num=samples_x, endpoint=True)
    #     d_1 = starts[0] * np.ones(2000) - offset
    #     fast_x = np.concatenate((d_0, d_1))
    #     d_2 = np.linspace(start=starts[0] - offset, stop=ends[0] - offset, num=samples_x,
    #                       endpoint=True)
    #     dx = d_2[1] - d_2[0]
    #     d_extension = np.linspace(d_2[-1] + dx, d_2[-1] + offset, num=int(offset / dx), endpoint=True)
    #     d_2 = np.concatenate((d_2, d_extension))
    #     d_2[:3500] = np.linspace(start=starts[0] - offset, stop=d_2[3500] + 0.025, num=3500, endpoint=True)
    #     d_2[3500:] += 0.0025
    #     line = np.concatenate((d_2, d_1))
    #     for i in range(lnm):
    #         fast_x = np.concatenate((fast_x, line))
    #     slow_y = np.arange(starts[1], ends[1], step=self.piezo_steps[1])
    #     slow_y = slow_y[:lnm]
    #     slow_y = np.repeat(slow_y, line.shape[0])
    #     slow_y = np.concatenate((np.ones(samples_x + 2000) * starts[1], slow_y))
    #     return fast_x, slow_y, fv, samples_x, line, offset


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
