# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np

from minimiao import logger


class TriggerSequence:

    def __init__(self, galvo=None, sample_rate=2.0e5, logg=None):
        self.logg = logg or logger.setup_logging()
        # daq
        self.sample_rate = sample_rate  # Hz
        self.galvo = galvo
        # digital triggers
        digital_starts = [0.00000, 0.0003, 0.0003, 0.00025, 0.00025, 0.0009]
        digital_ends = [0.02020, 0.0008, 0.0008, 0.003, 0.003, 0.001]
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in digital_ends]
        # piezo scanner
        self.piezo_conv_factors = [10., 10., 10.]
        self.piezo_steps = [0.032, 0.032, 0.16]
        self.piezo_ranges = [0.16, 0.16, 0.8]
        self.piezo_positions = [30., 30., 30.]
        self.piezo_return_time = 0.064
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
        # galvo scanner
        self.galvo_ramp_time = 300  # ~300 us
        self.galvo_ramp_time_samples = round(self.galvo_ramp_time * 1e-6 * self.sample_rate)
        self.galvo_step_response = 400  # ~325 us
        self.galvo_step_response_samples = round(self.galvo_step_response * 1e-6 * self.sample_rate)
        self.galvo_return_time = 300  # ~500 us
        self.galvo_return_samples = round(self.galvo_return_time * 1e-6 * self.sample_rate)
        self.galvo_origins = [0.0, -0.1]  # V
        self.galvo_ranges = [0.1, 0.2]  # V
        self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
        self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
        self.galvo_steps = [0.00, 0.00]  # volts
        self.galvo_pos = [galvo_positions(_start, _stop, _step) for (_start, _stop, _step) in
                          zip(self.galvo_starts, self.galvo_stops, self.galvo_steps)]
        self.galvo_scan_pos = [dps.size for dps in self.galvo_pos]
        # TTL
        self.trigger_pulse_width = 0.00010  # s
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
        self.slm_sw_total_time = 2 * 5310.72e-6  # s
        self.slm_sw_total_samples = round(self.slm_sw_total_time * self.sample_rate)
        self.slm_sw_end_time = 5270.187e-6 + self.slm_sw_total_time / 2  # s
        self.slm_sw_end_samples = round(self.slm_sw_end_time * self.sample_rate)
        self.slm_sw_on_time = 2 * 5000.0e-6 + 310.72e-6  # s
        self.slm_sw_on_samples = round(self.slm_sw_on_time * self.sample_rate)
        # camera
        self.interval_time = 0.02  # s
        self.interval_samples = int(np.ceil(self.interval_time * self.sample_rate))
        self.transfer_time = 0.01  # s
        self.transfer_samples = int(np.ceil(self.transfer_time * self.sample_rate))
        self.initial_time = 0.000  # s
        self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        self.exposure_time = 0.040  # s
        self.exposure_samples = int(np.ceil(self.exposure_time * self.sample_rate))
        self.standby_time = 0.03893  # s
        self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        self.frame_time = 0.05  # s
        self.frame_samples = int(np.ceil(self.frame_time * self.sample_rate))
        # motor
        self.motor_jog = 15.0  # deg
        self.motor_jog_pulse = 0.001
        self.motor_jog_samples = int(np.ceil(self.motor_jog_pulse * self.sample_rate))
        self.motor_rot_time = 0.1 + self.motor_jog_pulse * 8
        self.motor_rot_samples = int(np.ceil(self.motor_rot_time * self.sample_rate))

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

    def update_galvo_scan_parameters(self, origins=None, ranges=None, steps=None, returns=None):
        original_values = {"galvo_origins": self.galvo_origins, "galvo_ranges": self.galvo_ranges,
                           "galvo_starts": self.galvo_starts, "galvo_stops": self.galvo_stops,
                           "galvo_steps": self.galvo_steps, "galvo_pos": self.galvo_pos,
                           "galvo_return_time": self.galvo_return_time, "galvo_step_response": self.galvo_step_response}
        try:
            if origins is not None:
                self.galvo_origins = origins
            if ranges is not None:
                self.galvo_ranges = ranges
            if steps is not None:
                self.galvo_steps = steps
            if returns is not None:
                self.galvo_ramp_time, self.galvo_return_time, self.galvo_step_response = returns
            self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_pos = [galvo_positions(_start, _stop, _step) for (_start, _stop, _step) in
                              zip(self.galvo_starts, self.galvo_stops, self.galvo_steps)]
            self.galvo_scan_pos = [dps.size for dps in self.galvo_pos]
            self.galvo_ramp_time_samples = round(self.galvo_ramp_time * 1e-6 * self.sample_rate)
            self.galvo_step_response_samples = round(self.galvo_step_response * 1e-6 * self.sample_rate)
            self.galvo_return_samples = round(self.galvo_return_time * 1e-6 * self.sample_rate)
            act_samples = self.digital_ends[0] - self.digital_starts[0]
            if act_samples > 0:
                self.set_galvo_scan(act_samples)
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Galvo scanning parameters reverted to original values.")
            return

    def set_galvo_scan(self, scan_samples):
        scan_samples = max(int(scan_samples), 1)
        self.galvo.set_params(sample_rate_hz=self.sample_rate,
                              start_s=self.galvo_starts,
                              pixel_s=[r_ / scan_samples for r_ in self.galvo_ranges],
                              pixels_lines=(scan_samples, 1),
                              t_ramp_us=self.galvo_ramp_time, t_flyback_us=self.galvo_return_time,
                              t_settle_us=self.galvo_step_response)

    def update_digital_parameters(self, digital_starts=None, digital_ends=None):
        if digital_starts is not None:
            self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in digital_starts]
        if digital_ends is not None:
            self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in digital_ends]

    def update_camera_parameters(self, initial_time=None, exposure_time=None, standby_time=None, frame_rate=None,
                                 interval_time=None, transfer_time=None):
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
        if interval_time is not None:
            self.interval_time = interval_time
            self.interval_samples = int(np.ceil(self.interval_time * self.sample_rate))
        if transfer_time is not None:
            self.transfer_time = transfer_time
            self.transfer_samples = int(np.ceil(self.transfer_time * self.sample_rate))

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
        digital_channels = [0, 2, 3]
        act_samples = self.digital_ends[0] - self.digital_starts[0]
        self.cycle_samples = act_samples + self.interval_samples + self.slm_total_samples + max(self.transfer_samples, self.standby_samples)
        self.cycle_time = self.cycle_samples / self.sample_rate
        digital_triggers = np.zeros((3, self.cycle_samples), dtype=np.uint8)
        if act_samples > 0:
            digital_triggers[0, :act_samples] = 1
        digital_triggers[1, act_samples + self.interval_samples:act_samples + self.interval_samples + self.trigger_pulse_samples] = 1
        digital_triggers[2, act_samples + self.interval_samples:act_samples + self.interval_samples + self.trigger_pulse_samples] = 1
        analog_triggers, analog_channels = self.generate_galvo_triggers(digital_triggers[0])
        return digital_triggers, digital_channels, analog_triggers, analog_channels

    def generate_widefield_scan(self, n):
        digital_triggers, digital_channels, analog_triggers, analog_channels = self.generate_digital_triggers(0, 0)
        digital_triggers = np.tile(digital_triggers, n)
        pos = self.piezo_scan_pos[2]
        digital_triggers = np.tile(digital_triggers, self.piezo_scan_pos[2])
        if analog_triggers is not None:
            piezo_sequence = np.repeat(self.piezo_scan_positions[2], digital_triggers.shape[1])
            piezo_sequence = shift_array(piezo_sequence, self.return_samples, fill=piezo_sequence[0], direction="backward")
            analog_triggers = np.tile(analog_triggers, self.piezo_scan_pos[2])
            analog_triggers = np.concatenate((analog_triggers, piezo_sequence), axis=1)
            analog_channels.append(2)
        else:
            analog_triggers = np.repeat(self.piezo_scan_positions[2], digital_triggers.shape[1])
            analog_triggers = shift_array(analog_triggers, self.return_samples, fill=analog_triggers[0], direction="backward")
            analog_channels = [2]
        return digital_triggers, digital_channels, analog_triggers, analog_channels, pos

    def generate_sim_triggers(self, nang=3 , nph=6, rot=True):
        act_samples = self.digital_ends[0] - self.digital_starts[0]
        self.cycle_samples = act_samples + self.interval_samples + self.slm_total_samples + max(self.transfer_samples, self.standby_samples)
        self.cycle_time = self.cycle_samples / self.sample_rate
        digital_triggers = np.zeros((3, self.cycle_samples), dtype=np.uint8)
        if act_samples > 0:
            digital_triggers[0, :act_samples] = 1
        digital_triggers[1, act_samples + self.interval_samples:act_samples + self.interval_samples + self.trigger_pulse_samples] = 1
        digital_triggers[2, act_samples + self.interval_samples:act_samples + self.interval_samples + self.trigger_pulse_samples] = 1
        digital_triggers = np.tile(digital_triggers, (1, nph))
        if rot:
            digital_channels = [0, 2, 3, 4, 5, 6]
            motor_triggers = np.ones((3, self.cycle_samples * nph), dtype=np.uint8)
            digital_triggers = np.concatenate((digital_triggers, motor_triggers), axis=0)
            rn = 1  # (90 // nang) // self.motor_jog
            motor_offset = max(self.motor_rot_samples * rn - max(self.transfer_samples, self.standby_samples), 0)
            if motor_offset > 0:
                digital_triggers = np.concatenate((digital_triggers, np.zeros((6, motor_offset), dtype=np.uint8)), axis=1)
            motor_start = max(self.motor_rot_samples * rn, max(self.transfer_samples, self.standby_samples))
            digital_triggers[3, -motor_start:] = 0  # move
            digital_triggers[4, -motor_start:] = 1
            digital_triggers[5, -motor_start:] = 1
            for n in range(rn):
                digital_triggers[5, -motor_start + n * self.motor_rot_samples + 5 * self.motor_jog_samples:-motor_start + n * self.motor_rot_samples + 6 * self.motor_jog_samples] = 0  # forward
            digital_triggers = np.tile(digital_triggers, (1, nang))
            # digital_triggers[3, -motor_start:] = 0  # move
            # digital_triggers[4, -motor_start:] = 1
            # digital_triggers[5, -motor_start:] = 1
            # for n in range(rn):
            #     digital_triggers[4, -motor_start + n * self.motor_rot_samples + 5 * self.motor_jog_samples:-motor_start + n * self.motor_rot_samples + 6 * self.motor_jog_samples] = 0  # backward
            # if nang > 2:
            #     motor_back_triggers = np.ones((6, self.motor_rot_samples * rn * (nang - 2)), dtype=np.uint8)
            #     motor_back_triggers[:4, :] = 0
            #     for n in range(rn * (nang - 2)):
            #         motor_back_triggers[4, n * self.motor_rot_samples + 5 * self.motor_jog_samples:n * self.motor_rot_samples + 6 * self.motor_jog_samples] = 0
            #     digital_triggers = np.concatenate((digital_triggers, motor_back_triggers), axis=1)
        else:
            digital_channels = [0, 2, 3]
        analog_triggers, analog_channels = self.generate_galvo_triggers(digital_triggers[0])
        return digital_triggers, digital_channels, analog_triggers, analog_channels

    def generate_sim_scan(self, nang=3 , nph=6, rot=True):
        digital_triggers, digital_channels, analog_triggers, analog_channels = self.generate_sim_triggers(nang, nph, rot)
        pos = self.piezo_scan_pos[2]
        if analog_triggers is not None:
            piezo_sequence = np.repeat(self.piezo_scan_positions[2], digital_triggers.shape[1])
            piezo_sequence = shift_array(piezo_sequence, self.return_samples, fill=piezo_sequence[0], direction="backward")
            analog_triggers = np.tile(analog_triggers, self.piezo_scan_pos[2])
            analog_triggers = np.concatenate((analog_triggers, piezo_sequence), axis=1)
            analog_channels.append(2)
        else:
            analog_channels = [2]
            analog_triggers = np.repeat(self.piezo_scan_positions[2], digital_triggers.shape[1])
            analog_triggers = shift_array(analog_triggers, self.return_samples, fill=analog_triggers[0], direction="backward")
            digital_triggers = np.tile(digital_triggers, self.piezo_scan_pos[2])
        return digital_triggers, digital_channels, analog_triggers, analog_channels, pos

    def generate_nlsim_triggers(self, nang=3 , nph=6, off=0, rot=True):
        if off == 0:
            self.slm_sw_total_time = 2 * 1310.72e-6  # s
            self.slm_sw_total_samples = round(self.slm_sw_total_time * self.sample_rate)
            self.slm_sw_end_time = 1270.187e-6 + self.slm_sw_total_time / 2  # s
            self.slm_sw_end_samples = round(self.slm_sw_end_time * self.sample_rate)
            self.slm_sw_on_time = 2 * 1000e-6 + 310.72e-6  # s
            self.slm_sw_on_samples = round(self.slm_sw_on_time * self.sample_rate)
        else:
            self.slm_sw_total_time = 2 * 5310.72e-6  # s
            self.slm_sw_total_samples = round(self.slm_sw_total_time * self.sample_rate)
            self.slm_sw_end_time = 5270.187e-6 + self.slm_sw_total_time / 2  # s
            self.slm_sw_end_samples = round(self.slm_sw_end_time * self.sample_rate)
            self.slm_sw_on_time = 2 * 5000.0e-6 + 310.72e-6  # s
            self.slm_sw_on_samples = round(self.slm_sw_on_time * self.sample_rate)
        act_samples = self.digital_ends[0] - self.digital_starts[0]
        self.cycle_samples = act_samples + self.slm_sw_total_samples + self.interval_samples + self.slm_total_samples + max(self.transfer_samples, self.standby_samples)
        self.cycle_time = self.cycle_samples / self.sample_rate
        digital_triggers = np.zeros((3, self.cycle_samples), dtype=np.uint8)
        if act_samples > 0:
            digital_triggers[0, :act_samples] = 1
        digital_triggers[1, act_samples:act_samples + self.trigger_pulse_samples] = 1
        digital_triggers[1, act_samples + self.slm_sw_total_samples + self.interval_samples:act_samples + self.slm_sw_total_samples + self.interval_samples + self.trigger_pulse_samples] = 1
        digital_triggers[2, act_samples + self.slm_sw_total_samples + self.interval_samples + self.slm_start_samples:act_samples + self.slm_sw_total_samples + self.interval_samples + self.slm_start_samples + self.trigger_pulse_samples] = 1
        digital_triggers = np.tile(digital_triggers, (1, nph))
        if rot:
            digital_channels = [0, 2, 3, 4, 5, 6]
            motor_triggers = np.ones((3, self.cycle_samples * nph), dtype=np.uint8)
            digital_triggers = np.concatenate((digital_triggers, motor_triggers), axis=0)
            rn = 1  # (90 // nang) // self.motor_jog
            motor_offset = max(self.motor_rot_samples * rn - max(self.transfer_samples, self.standby_samples), 0)
            if motor_offset > 0:
                digital_triggers = np.concatenate((digital_triggers, np.zeros((6, motor_offset), dtype=np.uint8)), axis=1)
            motor_start = max(self.motor_rot_samples * rn, max(self.transfer_samples, self.standby_samples))
            digital_triggers[3, -motor_start:] = 0  # move
            digital_triggers[4, -motor_start:] = 1
            digital_triggers[5, -motor_start:] = 1
            for n in range(rn):
                digital_triggers[5, -motor_start + n * self.motor_rot_samples + 5 * self.motor_jog_samples:-motor_start + n * self.motor_rot_samples + 6 * self.motor_jog_samples] = 0  # forward
            digital_triggers = np.tile(digital_triggers, (1, nang))
            # digital_triggers[3, -motor_start:] = 0  # move
            # digital_triggers[4, -motor_start:] = 1
            # digital_triggers[5, -motor_start:] = 1
            # for n in range(rn):
            #     digital_triggers[4, -motor_start + n * self.motor_rot_samples + 5 * self.motor_jog_samples:-motor_start + n * self.motor_rot_samples + 6 * self.motor_jog_samples] = 0  # backward
            # if nang > 2:
            #     motor_back_triggers = np.ones((6, self.motor_rot_samples * rn * (nang - 2)), dtype=np.uint8)
            #     motor_back_triggers[:4, :] = 0
            #     for n in range(rn * (nang - 2)):
            #         motor_back_triggers[4, n * self.motor_rot_samples + 5 * self.motor_jog_samples:n * self.motor_rot_samples + 6 * self.motor_jog_samples] = 0
            #     digital_triggers = np.concatenate((digital_triggers, motor_back_triggers), axis=1)
        else:
            digital_channels = [0, 2, 3]
        analog_triggers, analog_channels = self.generate_galvo_triggers(digital_triggers[0])
        return digital_triggers, digital_channels, analog_triggers, analog_channels

    def generate_galvo_scan(self, laser_ttl):
        ttl = np.asarray(laser_ttl).astype(bool)
        n = ttl.size
        edges = np.diff(np.concatenate(([0], ttl.astype(np.int8), [0])))
        on_starts = np.flatnonzero(edges == 1)
        on_ends = np.flatnonzero(edges == -1)
        if on_starts.size == 0:
            return None
        self.set_galvo_scan(on_ends[0] - on_starts[0])
        galvo_x, galvo_y, gate = self.galvo.build_scan()
        ramp_samples = int(np.argmax(gate))
        galvo_sequences = np.empty((2, n))
        galvo_sequences[0] = galvo_x[0]
        galvo_sequences[1] = galvo_y[0]
        for i0 in on_starts:
            idx = (np.arange(galvo_x.size) + i0 - ramp_samples) % n
            galvo_sequences[0, idx] = galvo_x
            galvo_sequences[1, idx] = galvo_y
        return galvo_sequences

    def generate_galvo_triggers(self, laser_ttl):
        galvo_sequences = self.generate_galvo_scan(laser_ttl)
        if galvo_sequences is None:
            return None, None
        else:
            return galvo_sequences, [0, 1]

def galvo_positions(start, stop, step):
    if step == 0:
        return np.array([start])
    return np.arange(start, stop, step)


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
