import numpy as np


class TriggerSequence:
    class TriggerParameters:
        def __init__(self, sample_rate=250000):
            # daq
            self.sample_rate = sample_rate  # Hz
            # digital triggers
            self.laser_ports = {"405": 0, "488_0": 1, "488_1": 2, "488_2": 3}
            self.digital_starts = [0.0000, 0.00012, 0.00012, 0.00064, 0.00064, 0.00064, 0.00064]
            self.digital_ends = [0.0001, 0.00062, 0.00062, 0.00074, 0.00074, 0.00074, 0.00074]
            self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
            self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
            # piezo scanner
            self.piezo_conv_factors = [10., 10., 10.]
            self.piezo_steps = [0.04, 0.04, 0.16]
            self.piezo_ranges = [0.2, 0.2, 0.0]
            self.piezo_positions = [20., 20., 20.]
            self.piezo_return_time = 0.08
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
            # galvo switcher
            self.galvo_sw_settle = 0.00064  # s
            self.galvo_sw_settle_samples = int(np.ceil(self.galvo_sw_settle * self.sample_rate))
            self.galvo_sw_states = [4., -2., 0.]
            # galvo scanner
            self.galvo_step_response = int(3.2e-4 * self.sample_rate)  # ~320 us
            self.galvo_return = int(8e-4 * self.sample_rate)  # ~800 us
            self.ramp_down_fraction = 0.02
            self.ramp_down_offset = 20  # samples
            # galvo scan for read out
            self.galvo_origins = [0.0, 0.0]  # V
            self.galvo_ranges = [1.0, 1.0]  # V
            self.galvo_offsets = [0.008, 0.008]  # V
            self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.dot_ranges = [0.8, 0.8]  # V
            self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.dot_ranges)]
            self.dot_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.dot_ranges)]
            self.dot_step_s = 51  # samples
            self.dot_step_v = 0.0186  # volts
            self.dot_step_y = 0.0186  # volts
            self.up_rate = self.dot_step_v / self.dot_step_s
            self.dot_pos = np.arange(self.dot_starts[0], self.galvo_stops[0], self.dot_step_v)
            # sawtooth wave for read out
            self.ramp_up = np.arange(self.galvo_starts[0], self.galvo_stops[0] + self.dot_step_v, self.up_rate)
            self.ramp_up_samples = self.ramp_up.size
            self.ramp_down_samples = int(np.ceil(self.ramp_up_samples * self.ramp_down_fraction))
            self.frequency = int(self.sample_rate / self.ramp_up_samples)  # Hz
            # square wave for read out
            self.samples_high = 1
            self.samples_low = self.dot_step_s - self.samples_high
            self.samples_delay = int(np.abs(self.dot_starts[0] - self.galvo_starts[0]) / self.up_rate)
            self.samples_offset = self.ramp_up_samples - (self.samples_delay + self.dot_step_s * self.dot_pos.size)
            # galvo scan for activation
            self.galvo_origins_act = [0.00, 0.00]  # V
            self.galvo_ranges_act = [1.0, 1.0]  # V
            self.galvo_offsets_act = [0.008, 0.008]  # V
            self.galvo_starts_act = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins_act, self.galvo_ranges_act)]
            self.dot_ranges_act = [0.8, 0.8]  # V
            self.galvo_stops_act = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins_act, self.dot_ranges_act)]
            self.dot_starts_act = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins_act, self.dot_ranges_act)]
            self.dot_step_s_act = 41  # samples
            self.dot_step_v_act = 0.0185  # volts
            self.dot_step_y_act = 0.0185  # volts
            self.up_rate_act = self.dot_step_v_act / self.dot_step_s_act
            self.dot_pos_act = np.arange(self.dot_starts_act[0], self.galvo_stops_act[0], self.dot_step_v_act)
            # sawtooth wave for activation
            self.ramp_up_act = np.arange(self.galvo_starts_act[0], self.galvo_stops_act[0] + self.dot_step_v_act,
                                         self.up_rate_act)
            self.ramp_up_samples_act = self.ramp_up_act.size
            self.ramp_down_samples_act = int(np.ceil(self.ramp_up_samples_act * self.ramp_down_fraction))
            self.frequency_act = int(self.sample_rate / self.ramp_up_samples_act)  # Hz
            # square wave for activation
            self.samples_high_act = 1
            self.samples_low_act = self.dot_step_s_act - self.samples_high_act
            self.samples_delay_act = int(np.abs(self.dot_starts_act[0] - self.galvo_starts_act[0]) / self.up_rate_act)
            self.samples_offset_act = self.ramp_up_samples_act - self.samples_delay_act - self.dot_step_s_act * self.dot_pos_act.size
            # emccd camera
            self.cycle_time = 0.0021  # s
            self.initial_time = 0.00159  # s
            self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
            self.standby_time = 0.00171  # s
            self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
            self.exposure_samples = 0.0001  # s
            self.exposure_time = self.exposure_samples / self.sample_rate
            self.trigger_pulse_width = 50e-6  # s
            self.trigger_pulse_samples = int(np.ceil(self.trigger_pulse_width * self.sample_rate))
            # rolling shutter camera (light sheet mode of Hamamatsu sCMOS)
            self.line_interval = 1e-5  # s
            self.line_interval_samples = int(np.ceil(self.line_interval * self.sample_rate))
            self.trigger_delay_samples = 9 * self.line_interval_samples
            self.interval_line_number = 10.5
            self.line_exposure = 1e-5  # s
            self.line_exposure_samples = int(np.ceil(self.line_exposure * self.sample_rate))
            self.readout_timing = 0.001  # s
            self.readout_timing_samples = int(np.ceil(self.readout_timing * self.sample_rate))

    def __init__(self, bus, logg=None):
        self.bus = bus
        self.logg = logg or self.setup_logging()
        self._parameters = self.TriggerParameters()

    def __getattr__(self, item):
        if hasattr(self._parameters, item):
            return getattr(self._parameters, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def update_nidaq_parameters(self, sample_rate=None):
        if sample_rate is not None:
            self._parameters = self.TriggerParameters(sample_rate)

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
                self.return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
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

    def update_galvo_scan_parameters(self, origins=None, ranges=None, foci=None, offsets=None,
                                     origins_act=None, ranges_act=None, foci_act=None, offsets_act=None, sws=None):
        original_values = {"frequency": self.frequency, "galvo_origins": self.galvo_origins,
                           "galvo_ranges": self.galvo_ranges, "galvo_starts": self.galvo_starts,
                           "galvo_stops": self.galvo_stops, "galvo_offset": self.galvo_offsets,
                           "dot_ranges": self.dot_ranges, "dot_starts": self.dot_starts, "dot_step_v": self.dot_step_v,
                           "dot_step_s": self.dot_step_s, "dot_step_y": self.dot_step_y, "dot_pos": self.dot_pos,
                           "samples_low": self.samples_low, "samples_delay": self.samples_delay,
                           "samples_offset": self.samples_offset,
                           "frequency_act": self.frequency_act, "galvo_origins_act": self.galvo_origins_act,
                           "galvo_ranges_act": self.galvo_ranges_act, "galvo_starts_act": self.galvo_starts_act,
                           "galvo_stops_act": self.galvo_stops_act, "galvo_offset_act": self.galvo_offsets_act,
                           "dot_ranges_act": self.dot_ranges_act, "dot_starts_act": self.dot_starts_act,
                           "dot_step_v_act": self.dot_step_v_act, "dot_step_s_act": self.dot_step_s_act,
                           "dot_step_y_act": self.dot_step_y_act, "dot_pos_act": self.dot_pos_act,
                           "samples_low_act": self.samples_low_act, "samples_delay_act": self.samples_delay_act,
                           "samples_offset_act": self.samples_offset_act,
                           "galvo_sw_states": self.galvo_sw_states}
        try:
            if origins is not None:
                self.galvo_origins = origins
            if ranges is not None:
                self.galvo_ranges, self.dot_ranges = ranges
            if foci is not None:
                [self.dot_step_s, self.dot_step_v, self.dot_step_y] = foci
            if offsets is not None:
                self.galvo_offsets = offsets
            self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.dot_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.dot_ranges)]
            self.dot_pos = np.arange(self.dot_starts[0], self.dot_starts[0] + self.dot_ranges[0] + self.dot_step_v,
                                     self.dot_step_v)
            self.up_rate = self.dot_step_v / self.dot_step_s
            self.samples_low = self.dot_step_s - self.samples_high
            self.ramp_up = np.arange(self.galvo_starts[0], self.galvo_stops[0], self.up_rate)
            self.ramp_up_samples = self.ramp_up.size

            self.ramp_down_samples = int(np.ceil(self.ramp_up_samples * self.ramp_down_fraction))
            self.frequency = int(self.sample_rate / self.ramp_up_samples)  # Hz
            self.samples_delay = int(np.abs(self.dot_starts[0] - self.galvo_starts[0]) / self.up_rate)
            self.samples_offset = self.ramp_up_samples - self.samples_delay - self.dot_step_s * self.dot_pos.size
            if self.samples_offset < 0:
                self.logg.error("Invalid parameter combination leading to negative samples_offset.")
                raise ValueError("Invalid Galvo scanning parameters.")

            if origins_act is not None:
                self.galvo_origins_act = origins_act
            if ranges is not None:
                self.galvo_ranges_act, self.dot_ranges_act = ranges_act
            if foci is not None:
                [self.dot_step_s_act, self.dot_step_v_act, self.dot_step_y_act] = foci_act
            if offsets_act is not None:
                self.galvo_offsets_act = offsets_act
            self.galvo_starts_act = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins_act, self.galvo_ranges_act)]
            self.galvo_stops_act = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins_act, self.galvo_ranges_act)]
            self.dot_starts_act = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins_act, self.dot_ranges_act)]
            self.dot_pos_act = np.arange(self.dot_starts_act[0],
                                         self.dot_starts_act[0] + self.dot_ranges_act[0] + self.dot_step_v_act,
                                         self.dot_step_v_act)
            self.up_rate_act = self.dot_step_v_act / self.dot_step_s_act
            self.samples_low_act = self.dot_step_s_act - self.samples_high_act
            self.ramp_up_act = np.arange(self.galvo_starts_act[0], self.galvo_stops_act[0], self.up_rate_act)
            self.ramp_up_samples_act = self.ramp_up_act.size
            self.ramp_down_samples_act = int(np.ceil(self.ramp_up_samples_act * self.ramp_down_fraction))
            self.frequency_act = int(self.sample_rate / self.ramp_up_samples_act)  # Hz
            self.samples_delay_act = int(np.abs(self.dot_starts_act[0] - self.galvo_starts_act[0]) / self.up_rate_act)
            self.samples_offset_act = self.ramp_up_samples_act - self.samples_delay_act - self.dot_step_s_act * self.dot_pos_act.size
            if self.samples_offset_act < 0:
                self.logg.error("Invalid parameter combination leading to negative samples_offset.")
                raise ValueError("Invalid Galvo scanning parameters.")

            if sws is not None:
                self.galvo_sw_states = sws
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Galvo scanning parameters reverted to original values.")
            return

    def update_digital_parameters(self, digital_starts=None, digital_ends=None):
        if digital_starts is not None:
            self.digital_starts = digital_starts
        if digital_ends is not None:
            self.digital_ends = digital_ends
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]

    def update_camera_parameters(self, initial_time=None, standby_time=None, cycle_time=None):
        if initial_time is not None:
            self.initial_time = initial_time
            self.initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        if standby_time is not None:
            self.standby_time = standby_time
            self.standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        if self.cycle_time is not None:
            self.cycle_time = cycle_time

    def generate_digital_triggers(self, laser_list, camera, slm_seq=None):
        lasers = [self.laser_ports[las] for las in laser_list]
        cam_ind = camera + 4
        digital_channels = lasers.copy()
        digital_channels.append(cam_ind)
        interval_samples = max(self.initial_samples, self.galvo_sw_settle_samples)
        if slm_seq is None:
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
            switch_trigger = self.galvo_sw_states[camera] * np.ones(cycle_samples, dtype=np.float16)
            switch_trigger[:self.digital_starts[cam_ind] - self.galvo_sw_settle_samples] = self.galvo_sw_states[2]
            switch_trigger[self.digital_ends[cam_ind] + 1:] = self.galvo_sw_states[2]
        else:
            act_seq, cam_seq, self.exposure_time, self.exposure_samples = self.generate_slm_triggers(slm_seq)
            dark_samples = int(act_seq.shape[0] / 2)
            offset_samples = max(self.standby_samples - dark_samples, 0)
            if len(digital_channels) == 2:
                cycle_samples = interval_samples + act_seq.shape[0] + offset_samples
                digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
                digital_trigger[0][interval_samples:interval_samples + act_seq.shape[0]] = act_seq
                digital_trigger[1][interval_samples:interval_samples + cam_seq.shape[0]] = cam_seq
                switch_trigger = self.galvo_sw_states[2] * np.ones(cycle_samples, dtype=np.float16)
                switch_trigger[interval_samples:interval_samples + cam_seq.shape[0]] = self.galvo_sw_states[camera]
            elif len(digital_channels) == 3:
                expo_samples = self.digital_ends[lasers[0]] - self.digital_starts[lasers[0]]
                interp_samples = max(interval_samples, expo_samples)
                cycle_samples = interp_samples + act_seq.shape[0] + offset_samples
                digital_trigger = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
                digital_trigger[0][:expo_samples] = 1
                digital_trigger[1][interp_samples:interp_samples + act_seq.shape[0]] = act_seq
                digital_trigger[2][interp_samples:interp_samples + cam_seq.shape[0]] = cam_seq
                switch_trigger = self.galvo_sw_states[2] * np.ones(cycle_samples, dtype=np.float16)
                switch_trigger[interp_samples:interp_samples + cam_seq.shape[0]] = self.galvo_sw_states[camera]
            else:
                self.logg.error("Digital channels error.")
                raise ValueError("Digital channels number is wrong.")
        return digital_trigger, switch_trigger, digital_channels

    def generate_slm_triggers(self, slm_seq="5ms_dark_pair"):
        if slm_seq == "400us_lit_balanced":
            samps_total = round(776.64e-6 * self.sample_rate)
            expo_on = 400e-6
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(720.96e-6 * self.sample_rate)
            act_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            act_seq[:samps_total] = 1
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        elif slm_seq == "600us_lit_balanced":
            samps_total = round(976.747e-6 * self.sample_rate)
            expo_on = 600.107e-6
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(921.067e-6 * self.sample_rate)
            act_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            act_seq[:samps_total] = 1
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        elif slm_seq == "5ms_dark_pair":
            samps_total = round(5.31072e-3 * self.sample_rate)
            expo_on = 5.0e-3
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(5.270187e-3 * self.sample_rate)
            act_seq = np.ones(samps_total * 2, dtype=np.uint8)
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        elif slm_seq == "10ms_dark_pair":
            samps_total = round(10.31072e-3 * self.sample_rate)
            expo_on = 10.0e-3
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(10.270187e-3 * self.sample_rate)
            act_seq = np.ones(samps_total * 2, dtype=np.uint8)
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        elif slm_seq == "20ms_dark_pair":
            samps_total = round(20.31072e-3 * self.sample_rate)
            expo_on = 20.0e-3
            samps_on = round(expo_on * self.sample_rate)
            samps_start = int(270.187e-6 * self.sample_rate)
            samps_end = round(20.270187e-3 * self.sample_rate)
            act_seq = np.ones(samps_total * 2, dtype=np.uint8)
            cam_seq = np.zeros(samps_total * 2, dtype=np.uint8)
            cam_seq[samps_start:samps_end] = 1
        else:
            self.logg.error("SLM sequence error.")
            raise ValueError("SLM sequence is wrong.")
        return act_seq, cam_seq, expo_on, samps_on

    def generate_digital_triggers_for_scan(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        digital_triggers, switch_trigger, chs = self.generate_digital_triggers(lasers, camera)
        if self.standby_samples > self.return_samples:
            cycle_samples = digital_triggers.shape[1]
        else:
            compensate_samples = self.return_samples - self.standby_samples
            cycle_samples = digital_triggers.shape[1] + compensate_samples
            compensate_sequence = np.zeros((digital_triggers.shape[0], compensate_samples))
            digital_triggers = np.concatenate((digital_triggers, compensate_sequence), axis=1)
            compensate_sequence = switch_trigger[-1] * np.ones(compensate_samples)
            switch_trigger = np.concatenate((switch_trigger, compensate_sequence))
        return digital_triggers, switch_trigger, cycle_samples, chs

    def generate_piezo_scan(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        digital_triggers, switch_trigger, cycle_samples, dig_chs = self.generate_digital_triggers_for_scan(lasers, camera)
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
            switch_trigger = np.tile(switch_trigger, self.piezo_scan_pos[pch])
        return digital_triggers, switch_trigger, convert_list(piezo_sequences), dig_chs, pz_chs, pos

    def generate_piezo_line_scan(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        digital_triggers, switch_trigger, cycle_samples, dig_chs = self.generate_digital_triggers_for_scan(lasers, camera)
        pz_chs = []
        for i in range(3):
            if self.piezo_scan_pos[i] > 0:
                pz_chs.append(i)
        piezo_sequences = np.ones((len(pz_chs), cycle_samples))
        n = cycle_samples - self.return_samples
        for i, pch in enumerate(pz_chs):
            piezo_sequences[i] *= self.piezo_starts[pch]
            piezo_sequences[i, :n] = np.linspace(start=self.piezo_starts[pch],
                                                 stop=self.piezo_starts[pch] + self.piezo_ranges[pch],
                                                 num=n, endpoint=True)
        return digital_triggers, switch_trigger, piezo_sequences, dig_chs, pz_chs

    def generate_piezo_point_scan_2d(self, laser_list, camera, span="1 um"):
        lasers = [self.laser_ports[las] for las in laser_list]
        ramp_libs = {"1 um": self.generate_piezo_scan_ramp_1um(),
                     "2 um": self.generate_piezo_scan_ramp_2um()}
        fast_x, slow_y, fv, samples_x, line, offset = ramp_libs[span]
        # digital TTL
        cam_sw = self.galvo_sw_states[camera]
        cam_ind = camera + 4
        digital_channels = lasers.copy()
        initial_offset = max(self.initial_samples, self.galvo_sw_settle_samples)
        if initial_offset > self.digital_starts[cam_ind]:
            initial_offset -= self.digital_starts[cam_ind]
            self.digital_starts = [(_start + initial_offset) for _start in self.digital_starts]
            self.digital_ends = [(_end + initial_offset) for _end in self.digital_ends]
        self.exposure_samples = self.digital_ends[cam_ind] - self.digital_starts[cam_ind]
        self.exposure_time = self.exposure_samples / self.sample_rate
        interval_samples = int(self.sample_rate * self.piezo_steps[0] * 10 / fv)
        if interval_samples < self.standby_samples + self.exposure_samples:
            raise Exception("Error: step size is less than the sum of camera exposure and readout")
        cycle_samples = self.digital_ends[cam_ind] + interval_samples + 2
        digital_channels.append(cam_ind)
        digital_sequences = np.zeros((len(digital_channels), cycle_samples), dtype=np.uint8)
        for ln, chl in enumerate(digital_channels):
            digital_sequences[ln, self.digital_starts[chl]:self.digital_ends[chl]] = 1
        switch_galvo = np.ones(cycle_samples) * cam_sw
        switch_galvo[:self.digital_starts[cam_ind] - self.galvo_sw_settle_samples] = self.galvo_sw_states[2]
        switch_galvo[self.digital_ends[cam_ind] + 1:] = self.galvo_sw_states[2]
        digital_sequences = np.tile(digital_sequences, 32)
        switch_galvo = np.tile(switch_galvo, 32)
        offset_samples = np.zeros((len(digital_channels), int((0.025 + offset) / (0.2 / samples_x))))
        digital_sequences = np.concatenate((offset_samples, digital_sequences), axis=1)
        offset_samples = self.galvo_sw_states[2] * np.ones(int((0.025 + offset) / (0.2 / samples_x)))
        switch_galvo = np.concatenate((offset_samples, switch_galvo))
        offset_samples = np.zeros((len(digital_channels), line.shape[0] - digital_sequences.shape[1]))
        digital_sequences = np.concatenate((digital_sequences, offset_samples), axis=1)
        offset_samples = self.galvo_sw_states[2] * np.ones(line.shape[0] - switch_galvo.shape[0])
        switch_galvo = np.concatenate((switch_galvo, offset_samples))
        digital_sequences = np.tile(digital_sequences, 32)
        switch_galvo = np.tile(switch_galvo, 32)
        offset_samples = np.zeros((len(digital_channels), samples_x + 2000))
        digital_sequences = np.concatenate((offset_samples, digital_sequences), axis=1)
        offset_samples = self.galvo_sw_states[2] * np.ones(samples_x + 2000)
        switch_galvo = np.concatenate((offset_samples, switch_galvo))
        return np.vstack((fast_x, slow_y)), switch_galvo, digital_sequences, digital_channels, [0, 1], [2], 32 * 32

    def generate_piezo_scan_ramp_1um(self):
        lt = 0.25
        lnm = 32
        fv = 2 / lt
        starts = [s - 0.1 for s in self.piezo_positions]
        starts[1] += 0.025
        ends = [s + 0.1 for s in self.piezo_positions]
        offset = 0.03
        samples_x = int(lt * self.sample_rate)
        d_0 = np.linspace(start=starts[0], stop=ends[0], num=samples_x, endpoint=True)
        d_1 = starts[0] * np.ones(2000) - offset
        fast_x = np.concatenate((d_0, d_1))
        d_2 = np.linspace(start=starts[0] - offset, stop=ends[0] - offset, num=samples_x,
                          endpoint=True)
        dx = d_2[1] - d_2[0]
        d_extension = np.linspace(d_2[-1] + dx, d_2[-1] + offset, num=int(offset / dx), endpoint=True)
        d_2 = np.concatenate((d_2, d_extension))
        d_2[:3500] = np.linspace(start=starts[0] - offset, stop=d_2[3500] + 0.025, num=3500, endpoint=True)
        d_2[3500:] += 0.0025
        line = np.concatenate((d_2, d_1))
        for i in range(lnm):
            fast_x = np.concatenate((fast_x, line))
        slow_y = np.arange(starts[1], ends[1], step=self.piezo_steps[1])
        slow_y = slow_y[:lnm]
        slow_y = np.repeat(slow_y, line.shape[0])
        slow_y = np.concatenate((np.ones(samples_x + 2000) * starts[1], slow_y))
        return fast_x, slow_y, fv, samples_x, line, offset

    def generate_piezo_scan_ramp_2um(self):
        lt = 0.25
        lnm = 32
        fv = 2 / lt
        starts = [s - 0.1 for s in self.piezo_positions]
        starts[1] += 0.025
        ends = [s + 0.1 for s in self.piezo_positions]
        offset = 0.03
        samples_x = int(lt * self.sample_rate)
        d_0 = np.linspace(start=starts[0], stop=ends[0], num=samples_x, endpoint=True)
        d_1 = starts[0] * np.ones(2000) - offset
        fast_x = np.concatenate((d_0, d_1))
        d_2 = np.linspace(start=starts[0] - offset, stop=ends[0] - offset, num=samples_x,
                          endpoint=True)
        dx = d_2[1] - d_2[0]
        d_extension = np.linspace(d_2[-1] + dx, d_2[-1] + offset, num=int(offset / dx), endpoint=True)
        d_2 = np.concatenate((d_2, d_extension))
        d_2[:3500] = np.linspace(start=starts[0] - offset, stop=d_2[3500] + 0.025, num=3500, endpoint=True)
        d_2[3500:] += 0.0025
        line = np.concatenate((d_2, d_1))
        for i in range(lnm):
            fast_x = np.concatenate((fast_x, line))
        slow_y = np.arange(starts[1], ends[1], step=self.piezo_steps[1])
        slow_y = slow_y[:lnm]
        slow_y = np.repeat(slow_y, line.shape[0])
        slow_y = np.concatenate((np.ones(samples_x + 2000) * starts[1], slow_y))
        return fast_x, slow_y, fv, samples_x, line, offset

    def generate_digital_scanning_triggers(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        cam_sw = self.galvo_sw_states[camera]
        cam_ind = camera + 4
        lasers = lasers.copy()
        if 0 in lasers:
            # offset ramp for activation
            ramp_up_offset_act = np.linspace(0, self.galvo_offsets_act[0], self.ramp_up_samples_act + 1,
                                             dtype=np.float16, endpoint=True)
            ramp_down_offset_act = np.zeros(self.ramp_down_samples_act - 1, dtype=np.float16)
            ramp_offset_act = np.concatenate((ramp_up_offset_act, ramp_down_offset_act))
            slow_axis_offset_act = np.tile(ramp_offset_act, self.dot_pos_act.size)
            # galvo activation
            ramp_down_act = smooth_ramp(self.ramp_up_act[-1], self.ramp_up_act[0], self.ramp_down_samples_act)
            extended_cycle_act = np.concatenate((self.ramp_up_act, ramp_down_act))
            fast_axis_galvo_act = np.tile(extended_cycle_act, self.dot_pos_act.size)
            fast_axis_offset_act = np.linspace(0, self.galvo_offsets_act[1], fast_axis_galvo_act.size,
                                               dtype=np.float16, endpoint=True)
            fast_axis_galvo_act += np.repeat(fast_axis_offset_act[::extended_cycle_act.size], extended_cycle_act.size)
            slow_axis_galvo_act = np.zeros_like(fast_axis_galvo_act)
            indices_act = np.arange(self.ramp_up_samples_act + 1, len(fast_axis_galvo_act), extended_cycle_act.size)
            slow_axis_galvo_act[indices_act] = 1
            slow_axis_galvo_act = np.cumsum(slow_axis_galvo_act) * self.dot_step_y_act + self.dot_starts_act[
                1] + slow_axis_offset_act
            slow_axis_galvo_act[-self.ramp_down_samples_act:] = np.linspace(
                slow_axis_galvo_act[-self.ramp_down_samples_act], self.dot_starts_act[1], self.ramp_down_samples_act)
            fill_samples_act = max(0, self.galvo_sw_settle_samples - (
                    self.samples_offset_act + self.ramp_down_samples_act))
            fast_axis_galvo_act = np.pad(fast_axis_galvo_act, (self.galvo_return, fill_samples_act), 'constant',
                                         constant_values=(self.galvo_starts_act[0], self.galvo_starts_act[0]))
            slow_axis_galvo_act = np.pad(slow_axis_galvo_act, (self.galvo_return, fill_samples_act), 'constant',
                                         constant_values=(self.dot_starts_act[1], self.dot_starts_act[1]))
            _sqr_act = np.pad(np.ones(self.samples_high_act), (0, self.samples_low_act), 'constant',
                              constant_values=(0, 0))
            square_wave_act = np.pad(np.tile(_sqr_act, self.dot_pos_act.size),
                                     (self.samples_delay_act, self.samples_offset_act + self.ramp_down_samples_act),
                                     'constant', constant_values=(0, 0))
            laser_trigger_act = np.tile(square_wave_act, self.dot_pos_act.size - 1)
            laser_trigger_act = np.concatenate((np.zeros(square_wave_act.size), laser_trigger_act))
            if 3 in lasers:
                fast_axis_galvo_act[-fill_samples_act:] = self.galvo_starts[0]
                slow_axis_galvo_act[-fill_samples_act:] = self.dot_starts[0]
                laser_trigger_act = np.pad(laser_trigger_act, (self.galvo_return, fill_samples_act), 'constant',
                                           constant_values=(0, 0))
                switch_galvo_act = np.zeros(laser_trigger_act.shape)
                camera_trigger_act = np.zeros(laser_trigger_act.shape)
            else:
                switch_galvo_act = np.ones(fast_axis_galvo_act.shape) * cam_sw
                switch_galvo_act[:self.galvo_sw_settle_samples] = smooth_ramp(0., cam_sw, self.galvo_sw_settle_samples)
                switch_galvo_act[-self.galvo_sw_settle_samples:] = smooth_ramp(cam_sw, 0., self.galvo_sw_settle_samples)
                camera_trigger_act = np.ones(laser_trigger_act.shape, dtype=np.int8)
                camera_trigger_act[:self.samples_delay_act + square_wave_act.size] = 0
                camera_trigger_act[- self.samples_offset_act - self.ramp_down_samples_act:] = 0
                laser_trigger_act = np.pad(laser_trigger_act, (self.galvo_return, fill_samples_act), 'constant',
                                           constant_values=(0, 0))
                camera_trigger_act = np.pad(camera_trigger_act, (self.galvo_return, fill_samples_act), 'constant',
                                            constant_values=(0, 0))
                tl = self.samples_delay_act + self.galvo_sw_settle_samples + self.galvo_return
                self.exposure_samples = camera_trigger_act.shape[0] - tl
                self.exposure_time = self.exposure_samples / self.sample_rate
        # offset ramp
        ramp_up_offset = np.linspace(0, self.galvo_offsets[0], self.ramp_up_samples + 1, dtype=np.float16,
                                     endpoint=True)
        ramp_down_offset = np.zeros(self.ramp_down_samples - 1, dtype=np.float16)
        ramp_offset = np.concatenate((ramp_up_offset, ramp_down_offset))
        slow_axis_offset = np.tile(ramp_offset, self.dot_pos.size)
        # galvo read out
        ramp_down = smooth_ramp(self.ramp_up[-1], self.ramp_up[0], self.ramp_down_samples)
        extended_cycle = np.concatenate((self.ramp_up, ramp_down))
        fast_axis_galvo = np.tile(extended_cycle, self.dot_pos.size)
        fast_axis_offset = np.linspace(0, self.galvo_offsets[1], fast_axis_galvo.size, dtype=np.float16,
                                       endpoint=True)
        fast_axis_galvo += np.repeat(fast_axis_offset[::extended_cycle.size], extended_cycle.size)
        slow_axis_galvo = np.zeros_like(fast_axis_galvo)
        indices = np.arange(self.ramp_up_samples + 1, len(fast_axis_galvo), extended_cycle.size)
        slow_axis_galvo[indices] = 1
        slow_axis_galvo = np.cumsum(slow_axis_galvo) * self.dot_step_y + self.dot_starts[1] + slow_axis_offset
        slow_axis_galvo[-self.ramp_down_samples:] = np.linspace(slow_axis_galvo[-self.ramp_down_samples],
                                                                self.dot_starts[1], self.ramp_down_samples)
        fill_samples = max(0, self.galvo_sw_settle_samples - (self.samples_offset + self.ramp_down_samples))
        fast_axis_galvo = np.pad(fast_axis_galvo, (self.galvo_return, fill_samples), 'constant',
                                 constant_values=(self.galvo_starts[0], self.galvo_starts[0]))
        slow_axis_galvo = np.pad(slow_axis_galvo, (self.galvo_return, fill_samples), 'constant',
                                 constant_values=(self.dot_starts[1], self.dot_starts[1]))
        switch_galvo = np.ones(fast_axis_galvo.shape) * cam_sw
        switch_galvo[:self.galvo_sw_settle_samples] = smooth_ramp(0., cam_sw, self.galvo_sw_settle_samples)
        switch_galvo[-self.galvo_sw_settle_samples:] = smooth_ramp(cam_sw, 0., self.galvo_sw_settle_samples)
        _sqr = np.pad(np.ones(self.samples_high), (0, self.samples_low), 'constant', constant_values=(0, 0))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size),
                             (self.samples_delay, self.samples_offset + self.ramp_down_samples), 'constant',
                             constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size - 1)
        laser_trigger = np.concatenate((np.zeros(square_wave.size), laser_trigger))
        camera_trigger = np.ones(laser_trigger.shape, dtype=np.int8)
        camera_trigger[:self.samples_delay + square_wave.size] = 0
        camera_trigger[- self.samples_offset - self.ramp_down_samples:] = 0
        laser_trigger = np.pad(laser_trigger, (self.galvo_return, fill_samples), 'constant', constant_values=(0, 0))
        camera_trigger = np.pad(camera_trigger, (self.galvo_return, fill_samples), 'constant', constant_values=(0, 0))
        tl = self.samples_delay + self.galvo_sw_settle_samples + self.galvo_return
        self.exposure_samples = camera_trigger.shape[0] - tl
        self.exposure_time = self.exposure_samples / self.sample_rate
        # all
        digital_sequences = [np.empty((0,)) for _ in range(len(lasers) + 1)]
        galvo_sequences = [np.empty((0,)) for _ in range(3)]
        for _, las in enumerate(lasers):
            if las == 0:
                trig = laser_trigger_act
                gvf = fast_axis_galvo_act
                gvs = slow_axis_galvo_act
                sw = switch_galvo_act
                cm = camera_trigger_act
            elif las == 1:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                sw = np.zeros(trig.shape)
                cm = np.zeros(trig.shape)
            elif las == 2:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                sw = np.zeros(trig.shape)
                cm = np.zeros(trig.shape)
            elif las == 3:
                trig = laser_trigger
                gvf = fast_axis_galvo
                gvs = slow_axis_galvo
                sw = switch_galvo
                cm = camera_trigger
            if (las == 2) and (1 in lasers):
                for i in range(len(lasers)):
                    if lasers[i] == 1:
                        temp = digital_sequences[i]
                    if lasers[i] == las:
                        digital_sequences[i] = temp
            else:
                galvo_sequences[0] = np.append(galvo_sequences[0], gvf)
                galvo_sequences[1] = np.append(galvo_sequences[1], gvs)
                digital_sequences[-1] = np.append(digital_sequences[-1], cm)
                galvo_sequences[2] = np.append(galvo_sequences[2], sw)
                for i in range(len(lasers)):
                    if lasers[i] == las:
                        digital_sequences[i] = np.append(digital_sequences[i], trig)
                    else:
                        digital_sequences[i] = np.append(digital_sequences[i], np.zeros(trig.shape))
        lasers.append(cam_ind)
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.append(dtr, dtr[-1] * np.ones(self.standby_samples))
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.append(gtr, gtr[-1] * np.ones(self.standby_samples))
        return np.asarray(digital_sequences), np.asarray(galvo_sequences), lasers

    def generate_dotscan_resolft_2d(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        cam_sw = self.galvo_sw_states[camera]
        cam_ind = camera + 4
        lasers = lasers.copy()
        # read out galvo
        ramp_up_offset = np.linspace(0, self.galvo_offsets[0], self.ramp_up_samples + 1, dtype=np.float16,
                                     endpoint=True)
        ramp_down_offset = np.zeros(self.ramp_down_samples - 1, dtype=np.float16)
        ramp_offset = np.concatenate((ramp_up_offset, ramp_down_offset))
        slow_axis_offset = np.tile(ramp_offset, self.dot_pos.size)
        ramp_down = np.linspace(self.ramp_up[-1], self.ramp_up[0], num=self.ramp_down_samples, endpoint=True)
        extended_cycle = np.concatenate((self.ramp_up, ramp_down))
        fast_axis_galvo = np.tile(extended_cycle, self.dot_pos.size)
        fast_axis_offset = np.linspace(0, self.galvo_offsets[1], fast_axis_galvo.size, dtype=np.float16,
                                       endpoint=True)
        fast_axis_galvo += np.repeat(fast_axis_offset[::extended_cycle.size], extended_cycle.size)
        slow_axis_galvo = np.zeros_like(fast_axis_galvo)
        indices = np.arange(self.ramp_up_samples + 1, len(fast_axis_galvo), extended_cycle.size)
        slow_axis_galvo[indices] = 1
        slow_axis_galvo = np.cumsum(slow_axis_galvo) * self.dot_step_y + self.dot_starts[1] + slow_axis_offset
        slow_axis_galvo[-self.ramp_down_samples:] = np.linspace(slow_axis_galvo[-self.ramp_down_samples],
                                                                self.dot_starts[1], self.ramp_down_samples)
        fill_samples = max(0, self.galvo_sw_settle_samples - (self.samples_offset + self.ramp_down_samples))
        fast_axis_galvo = np.pad(fast_axis_galvo, (self.galvo_return, fill_samples), 'constant',
                                 constant_values=(self.galvo_starts[0], self.galvo_starts[0]))
        slow_axis_galvo = np.pad(slow_axis_galvo, (self.galvo_return, fill_samples), 'constant',
                                 constant_values=(self.dot_starts[1], self.dot_starts[1]))
        _sqr = np.pad(np.ones(self.samples_high), (0, self.samples_low), 'constant', constant_values=(0, 0))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size),
                             (self.samples_delay, self.samples_offset + self.ramp_down_samples), 'constant',
                             constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size - 1)
        laser_trigger = np.concatenate((np.zeros(square_wave.size), laser_trigger))
        camera_trigger = np.ones(laser_trigger.shape, dtype=np.int8)
        camera_trigger[:self.samples_delay + square_wave.size] = 0
        camera_trigger[- self.samples_offset - self.ramp_down_samples:] = 0
        laser_trigger = np.pad(laser_trigger, (self.galvo_return, fill_samples), 'constant', constant_values=(0, 0))
        camera_trigger = np.pad(camera_trigger, (self.galvo_return, fill_samples), 'constant', constant_values=(0, 0))
        tl = self.samples_delay + self.galvo_sw_settle_samples + self.galvo_return
        self.exposure_samples = camera_trigger.shape[0] - tl
        self.exposure_time = self.exposure_samples / self.sample_rate
        # activation galvo
        ramp_up_offset_act = np.linspace(0, self.galvo_offsets_act[0], self.ramp_up_samples_act + 1,
                                         dtype=np.float16, endpoint=True)
        ramp_down_offset_act = np.zeros(self.ramp_down_samples_act - 1, dtype=np.float16)
        ramp_offset_act = np.concatenate((ramp_up_offset_act, ramp_down_offset_act))
        slow_axis_offset_act = np.tile(ramp_offset_act, self.dot_pos_act.size)
        ramp_down_act = np.linspace(self.ramp_up_act[-1], self.ramp_up_act[0], num=self.ramp_down_samples_act,
                                    endpoint=True)
        extended_cycle_act = np.concatenate((self.ramp_up_act, ramp_down_act))
        fast_axis_galvo_act = np.tile(extended_cycle_act, self.dot_pos_act.size)
        fast_axis_offset_act = np.linspace(0, self.galvo_offsets_act[1], fast_axis_galvo_act.size,
                                           dtype=np.float16, endpoint=True)
        fast_axis_galvo_act += np.repeat(fast_axis_offset_act[::extended_cycle_act.size], extended_cycle_act.size)
        slow_axis_galvo_act = np.zeros_like(fast_axis_galvo_act)
        indices_act = np.arange(self.ramp_up_samples_act + 1, len(fast_axis_galvo_act), extended_cycle_act.size)
        slow_axis_galvo_act[indices_act] = 1
        slow_axis_galvo_act = np.cumsum(slow_axis_galvo_act) * self.dot_step_y_act + self.dot_starts_act[
            1] + slow_axis_offset_act
        slow_axis_galvo_act[-self.ramp_down_samples_act:] = np.linspace(
            slow_axis_galvo_act[-self.ramp_down_samples_act], self.dot_starts_act[1], self.ramp_down_samples_act)
        fill_samples_act = max(0, self.galvo_sw_settle_samples - (self.samples_offset_act + self.ramp_down_samples_act))
        fast_axis_galvo_act = np.pad(fast_axis_galvo_act, (self.galvo_return, fill_samples_act), 'constant',
                                     constant_values=(self.galvo_starts_act[0], self.galvo_starts_act[0]))
        slow_axis_galvo_act = np.pad(slow_axis_galvo_act, (self.galvo_return, fill_samples_act), 'constant',
                                     constant_values=(self.dot_starts_act[1], self.dot_starts_act[1]))
        _sqr_act = np.pad(np.ones(self.samples_high_act), (0, self.samples_low_act), 'constant', constant_values=(0, 0))
        square_wave_act = np.pad(np.tile(_sqr_act, self.dot_pos_act.size),
                                 (self.samples_delay_act, self.samples_offset_act + self.ramp_down_samples_act),
                                 'constant', constant_values=(0, 0))
        laser_trigger_act = np.tile(square_wave_act, self.dot_pos_act.size - 1)
        laser_trigger_act = np.concatenate((np.zeros(square_wave_act.size), laser_trigger_act))
        fast_axis_galvo_act[-fill_samples_act:] = self.galvo_starts[0]
        slow_axis_galvo_act[-fill_samples_act:] = self.dot_starts[0]
        laser_trigger_act = np.pad(laser_trigger_act, (self.galvo_return, fill_samples_act), 'constant',
                                   constant_values=(0, 0))
        # switching galvo
        switch_galvo = np.ones(fast_axis_galvo.shape) * cam_sw
        switch_galvo[:self.galvo_sw_settle_samples] = smooth_ramp(0., cam_sw, self.galvo_sw_settle_samples)
        switch_galvo[-self.galvo_sw_settle_samples:] = smooth_ramp(cam_sw, 0., self.galvo_sw_settle_samples)
        # all
        digital_sequences = [np.empty((0,)) for _ in range(len(lasers) + 1)]
        galvo_sequences = [np.empty((0,)) for _ in range(3)]
        for _, las in enumerate(lasers):
            if las == 0:
                trig = laser_trigger_act
                gvf = fast_axis_galvo_act
                gvs = slow_axis_galvo_act
                sw = np.zeros(trig.shape)
                cm = np.zeros(trig.shape)
            elif las == 1:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                sw = np.zeros(trig.shape)
                cm = np.zeros(trig.shape)
            elif las == 2:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                sw = np.zeros(trig.shape)
                cm = np.zeros(trig.shape)
            elif las == 3:
                trig = laser_trigger
                gvf = fast_axis_galvo
                gvs = slow_axis_galvo
                sw = switch_galvo
                cm = camera_trigger
            if (las == 2) and (1 in lasers):
                for i in range(len(lasers)):
                    if lasers[i] == 1:
                        temp = digital_sequences[i]
                    if lasers[i] == las:
                        digital_sequences[i] = temp
            else:
                galvo_sequences[0] = np.append(galvo_sequences[0], gvf)
                galvo_sequences[1] = np.append(galvo_sequences[1], gvs)
                digital_sequences[-1] = np.append(digital_sequences[-1], cm)
                galvo_sequences[2] = np.append(galvo_sequences[2], sw)
                for i in range(len(lasers)):
                    if lasers[i] == las:
                        digital_sequences[i] = np.append(digital_sequences[i], trig)
                    else:
                        digital_sequences[i] = np.append(digital_sequences[i], np.zeros(trig.shape))
        lasers.append(cam_ind)
        idle_samples = max(self.standby_samples, self.return_samples)
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.append(dtr, dtr[-1] * np.ones(idle_samples))
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.append(gtr, gtr[-1] * np.ones(idle_samples))
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences[0] = np.repeat(self.piezo_scan_positions[0], digital_sequences[0].shape[0])
        piezo_sequences[0] = shift_array(piezo_sequences[0], idle_samples, piezo_sequences[0][0], "backward")
        piezo_sequences[0][-idle_samples:] = piezo_sequences[0][0]
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.tile(dtr, self.piezo_scan_pos[0])
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.tile(gtr, self.piezo_scan_pos[0])
        piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[1])
        piezo_sequences[1] = np.repeat(self.piezo_scan_positions[1], digital_sequences[0].shape[0])
        piezo_sequences[1] = shift_array(piezo_sequences[1], idle_samples, piezo_sequences[1][0], "backward")
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.tile(dtr, self.piezo_scan_pos[1])
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.tile(gtr, self.piezo_scan_pos[1])
        scan_pos = self.piezo_scan_pos[0] * self.piezo_scan_pos[1]
        self.logg.info("\nGalvo start, and stop: {} \n"
                       "Dot start, step, range, and numbers: {} \n"
                       "Piezo starts: {} \n"
                       "Piezo steps: {} \n"
                       "Piezo ranges: {} \n"
                       "Piezo positions: {}".format([self.galvo_starts, self.galvo_stops],
                                                    [self.dot_starts, [self.dot_step_v, self.dot_step_y],
                                                     self.dot_ranges, self.dot_pos.size], self.piezo_starts,
                                                    self.piezo_steps, self.piezo_ranges, scan_pos))
        return (np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences),
                lasers, scan_pos)

    def update_lightsheet_rolling(self, interval_line_number):
        self.line_interval_samples = int(np.ceil(((
                                                          self.samples_high + self.samples_low) * self.dot_pos.size + self.samples_delay + self.samples_offset + self.ramp_down_samples) / interval_line_number))
        self.line_exposure_samples = (self.samples_high + self.samples_low) * self.dot_pos.size
        self.trigger_delay_samples = 10 * self.line_interval_samples
        return self.line_exposure_samples / self.sample_rate, self.line_interval_samples / self.sample_rate

    def generate_digital_scanning_triggers_rolling(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        cam_ind = camera + 4
        lasers = lasers.copy()
        offset_samples = max(self.trigger_delay_samples + 2, self.galvo_return)
        if 0 in lasers:
            # galvo activation
            ramp_down_act = np.linspace(self.ramp_up_act[-1], self.ramp_up_act[0], num=self.ramp_down_samples_act,
                                        endpoint=True)
            extended_cycle_act = np.concatenate((self.ramp_up_act, ramp_down_act))
            fast_axis_galvo_act = np.tile(extended_cycle_act, self.dot_pos_act.size)
            slow_axis_galvo_act = np.zeros_like(fast_axis_galvo_act)
            indices_act = np.arange(self.ramp_up_samples_act + 1, len(fast_axis_galvo_act), extended_cycle_act.size)
            slow_axis_galvo_act[indices_act] = 1
            slow_axis_galvo_act = np.cumsum(slow_axis_galvo_act) * self.dot_step_y_act + self.dot_starts_act[1]
            slow_axis_galvo_act[-self.ramp_down_samples_act:] = np.linspace(
                slow_axis_galvo_act[-self.ramp_down_samples_act], self.dot_starts_act[1], self.ramp_down_samples_act)
            _sqr_act = np.pad(np.ones(self.samples_high_act), (0, self.samples_low_act), 'constant',
                              constant_values=(0, 0))
            square_wave_act = np.pad(np.tile(_sqr_act, self.dot_pos_act.size),
                                     (self.samples_delay_act, self.samples_offset_act + self.ramp_down_samples_act),
                                     'constant', constant_values=(0, 0))
            laser_trigger_act = np.tile(square_wave_act, self.dot_pos_act.size)
            if 3 in lasers:
                fast_axis_galvo_act = np.pad(fast_axis_galvo_act, (4, 4), 'constant',
                                             constant_values=(self.galvo_starts_act[0], self.galvo_starts[0]))
                slow_axis_galvo_act = np.pad(slow_axis_galvo_act, (4, 4), 'constant',
                                             constant_values=(self.dot_starts_act[1], self.dot_starts[0]))
                laser_trigger_act = np.pad(laser_trigger_act, (4, 4), 'constant', constant_values=(0, 0))
                camera_trigger_act = np.zeros(laser_trigger_act.shape)
            else:
                fast_axis_galvo_act = np.pad(fast_axis_galvo_act, (offset_samples, self.readout_timing_samples),
                                             'constant',
                                             constant_values=(self.galvo_starts_act[0], self.galvo_starts_act[0]))
                slow_axis_galvo_act = np.pad(slow_axis_galvo_act, (offset_samples, self.readout_timing_samples),
                                             'constant',
                                             constant_values=(self.dot_starts_act[1], self.dot_starts_act[1]))
                laser_trigger_act = np.pad(laser_trigger_act, (offset_samples, self.readout_timing_samples), 'constant',
                                           constant_values=(0, 0))
                camera_trigger_act = np.ones(laser_trigger_act.shape, dtype=np.int8)
                camera_trigger_act[:2] = 0
                camera_trigger_act[self.trigger_delay_samples - 2:] = 0
        # galvo read out
        ramp_down = np.linspace(self.ramp_up[-1], self.ramp_up[0], num=self.ramp_down_samples, endpoint=True)
        extended_cycle = np.concatenate((self.ramp_up, ramp_down))
        fast_axis_galvo = np.tile(extended_cycle, self.dot_pos.size)
        slow_axis_galvo = np.zeros_like(fast_axis_galvo)
        indices = np.arange(self.ramp_up_samples + 1, len(fast_axis_galvo), extended_cycle.size)
        slow_axis_galvo[indices] = 1
        slow_axis_galvo = np.cumsum(slow_axis_galvo) * self.dot_step_y + self.dot_starts[1]
        slow_axis_galvo[-self.ramp_down_samples:] = np.linspace(slow_axis_galvo[-self.ramp_down_samples],
                                                                self.dot_starts[1], self.ramp_down_samples)
        _sqr = np.pad(np.ones(self.samples_high), (0, self.samples_low), 'constant', constant_values=(0, 0))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size),
                             (self.samples_delay, self.samples_offset + self.ramp_down_samples), 'constant',
                             constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        fast_axis_galvo = np.pad(fast_axis_galvo, (offset_samples, self.readout_timing_samples), 'constant',
                                 constant_values=(self.galvo_starts[0], self.galvo_starts[0]))
        slow_axis_galvo = np.pad(slow_axis_galvo, (offset_samples, self.readout_timing_samples), 'constant',
                                 constant_values=(self.dot_starts[1], self.dot_starts[1]))
        laser_trigger = np.pad(laser_trigger, (offset_samples, self.readout_timing_samples), 'constant',
                               constant_values=(0, 0))
        camera_trigger = np.ones(laser_trigger.shape, dtype=np.int8)
        camera_trigger[:2] = 0
        camera_trigger[self.trigger_delay_samples - 2:] = 0
        # all
        digital_sequences = [np.empty((0,)) for _ in range(len(lasers) + 1)]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        for _, las in enumerate(lasers):
            if las == 0:
                trig = laser_trigger_act
                gvf = fast_axis_galvo_act
                gvs = slow_axis_galvo_act
                cm = camera_trigger_act
            if las == 1:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                cm = np.zeros(trig.shape)
            if las == 2:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                cm = np.zeros(trig.shape)
            if las == 3:
                trig = laser_trigger
                gvf = fast_axis_galvo
                gvs = slow_axis_galvo
                cm = camera_trigger
            if (las == 2) and (1 in lasers):
                for i in range(len(lasers)):
                    if lasers[i] == 1:
                        temp = digital_sequences[i]
                    if lasers[i] == las:
                        digital_sequences[i] = temp
            else:
                galvo_sequences[0] = np.append(galvo_sequences[0], gvf)
                galvo_sequences[1] = np.append(galvo_sequences[1], gvs)
                digital_sequences[-1] = np.append(digital_sequences[-1], cm)
                for i in range(len(lasers)):
                    if lasers[i] == las:
                        digital_sequences[i] = np.append(digital_sequences[i], trig)
                    else:
                        digital_sequences[i] = np.append(digital_sequences[i], np.zeros(trig.shape))
        lasers.append(cam_ind)
        return np.asarray(digital_sequences), np.asarray(galvo_sequences), lasers

    def generate_dotscan_resolft_2d_rolling(self, laser_list, camera):
        lasers = [self.laser_ports[las] for las in laser_list]
        cam_ind = camera + 4
        lasers = lasers.copy()
        # read out galvo
        ramp_down = np.linspace(self.ramp_up[-1], self.ramp_up[0], num=self.ramp_down_samples, endpoint=True)
        extended_cycle = np.concatenate((self.ramp_up, ramp_down))
        fast_axis_galvo = np.tile(extended_cycle, self.dot_pos.size)
        slow_axis_galvo = np.zeros_like(fast_axis_galvo)
        indices = np.arange(self.ramp_up_samples + 1, len(fast_axis_galvo), extended_cycle.size)
        slow_axis_galvo[indices] = 1
        slow_axis_galvo = np.cumsum(slow_axis_galvo) * self.dot_step_y + self.dot_starts[1]
        slow_axis_galvo[-self.ramp_down_samples:] = np.linspace(slow_axis_galvo[-self.ramp_down_samples],
                                                                self.dot_starts[1], self.ramp_down_samples)
        fill_samples = max(0, self.galvo_sw_settle_samples - (self.samples_offset + self.ramp_down_samples))
        fast_axis_galvo = np.pad(fast_axis_galvo, (self.galvo_return, fill_samples), 'constant',
                                 constant_values=(self.galvo_starts[0], self.galvo_starts[0]))
        slow_axis_galvo = np.pad(slow_axis_galvo, (self.galvo_return, fill_samples), 'constant',
                                 constant_values=(self.dot_starts[1], self.dot_starts[1]))
        _sqr = np.pad(np.ones(self.samples_high), (0, self.samples_low), 'constant', constant_values=(0, 0))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size),
                             (self.samples_delay, self.samples_offset + self.ramp_down_samples), 'constant',
                             constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        camera_trigger = np.ones(laser_trigger.shape, dtype=np.int8)
        camera_trigger[:self.samples_delay] = 0
        camera_trigger[- self.samples_offset - self.ramp_down_samples:] = 0
        laser_trigger = np.pad(laser_trigger, (self.galvo_return, fill_samples), 'constant', constant_values=(0, 0))
        camera_trigger = np.pad(camera_trigger, (self.galvo_return, fill_samples), 'constant', constant_values=(0, 0))
        tl = self.samples_delay + self.galvo_sw_settle_samples + self.galvo_return
        self.exposure_samples = camera_trigger.shape[0] - tl
        self.exposure_time = self.exposure_samples / self.sample_rate
        # activation galvo
        ramp_down_act = np.linspace(self.ramp_up_act[-1], self.ramp_up_act[0], num=self.ramp_down_samples_act,
                                    endpoint=True)
        extended_cycle_act = np.concatenate((self.ramp_up_act, ramp_down_act))
        fast_axis_galvo_act = np.tile(extended_cycle_act, self.dot_pos_act.size)
        slow_axis_galvo_act = np.zeros_like(fast_axis_galvo_act)
        indices_act = np.arange(self.ramp_up_samples_act + 1, len(fast_axis_galvo_act), extended_cycle_act.size)
        slow_axis_galvo_act[indices_act] = 1
        slow_axis_galvo_act = np.cumsum(slow_axis_galvo_act) * self.dot_step_y_act + self.dot_starts_act[1]
        slow_axis_galvo_act[-self.ramp_down_samples_act:] = np.linspace(
            slow_axis_galvo_act[-self.ramp_down_samples_act], self.dot_starts_act[1], self.ramp_down_samples_act)
        fill_samples_act = max(0, self.galvo_sw_settle_samples - (self.samples_offset_act + self.ramp_down_samples_act))
        fast_axis_galvo_act = np.pad(fast_axis_galvo_act, (self.galvo_return, fill_samples_act), 'constant',
                                     constant_values=(self.galvo_starts_act[0], self.galvo_starts_act[0]))
        slow_axis_galvo_act = np.pad(slow_axis_galvo_act, (self.galvo_return, fill_samples_act), 'constant',
                                     constant_values=(self.dot_starts_act[1], self.dot_starts_act[1]))
        _sqr_act = np.pad(np.ones(self.samples_high_act), (0, self.samples_low_act), 'constant', constant_values=(0, 0))
        square_wave_act = np.pad(np.tile(_sqr_act, self.dot_pos_act.size),
                                 (self.samples_delay_act, self.samples_offset_act + self.ramp_down_samples_act),
                                 'constant', constant_values=(0, 0))
        laser_trigger_act = np.tile(square_wave_act, self.dot_pos_act.size)
        fast_axis_galvo_act[-fill_samples_act:] = self.galvo_starts[0]
        slow_axis_galvo_act[-fill_samples_act:] = self.dot_starts[0]
        laser_trigger_act = np.pad(laser_trigger_act, (self.galvo_return, fill_samples_act), 'constant',
                                   constant_values=(0, 0))
        # all
        digital_sequences = [np.empty((0,)) for _ in range(len(lasers) + 1)]
        galvo_sequences = [np.empty((0,)) for _ in range(3)]
        for _, las in enumerate(lasers):
            if las == 0:
                trig = laser_trigger_act
                gvf = fast_axis_galvo_act
                gvs = slow_axis_galvo_act
                cm = np.zeros(trig.shape)
            elif las == 1:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                cm = np.zeros(trig.shape)
            elif las == 2:
                itl = int(np.ceil(0.0008 * self.sample_rate))
                trig = np.pad(np.ones(self.digital_ends[las] - self.digital_starts[las]), (itl, itl), 'constant',
                              constant_values=(0, 0))
                gvf = np.ones(trig.shape) * fast_axis_galvo[0]
                gvs = np.ones(trig.shape) * slow_axis_galvo[0]
                cm = np.zeros(trig.shape)
            elif las == 3:
                trig = laser_trigger
                gvf = fast_axis_galvo
                gvs = slow_axis_galvo
                cm = camera_trigger
            galvo_sequences[0] = np.append(galvo_sequences[0], gvf)
            galvo_sequences[1] = np.append(galvo_sequences[1], gvs)
            digital_sequences[-1] = np.append(digital_sequences[-1], cm)
            for i in range(len(lasers)):
                if lasers[i] == las:
                    digital_sequences[i] = np.append(digital_sequences[i], trig)
                else:
                    digital_sequences[i] = np.append(digital_sequences[i], np.zeros(trig.shape))
        lasers.append(cam_ind)
        idle_samples = max(self.standby_samples, self.return_samples)
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.append(dtr, dtr[-1] * np.ones(idle_samples))
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.append(gtr, gtr[-1] * np.ones(idle_samples))
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences[0] = np.repeat(self.piezo_scan_positions[0], digital_sequences[0].shape[0])
        piezo_sequences[0] = shift_array(piezo_sequences[0], idle_samples, piezo_sequences[0][0], "backward")
        piezo_sequences[0][-idle_samples:] = piezo_sequences[0][0]
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.tile(dtr, self.piezo_scan_pos[0])
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.tile(gtr, self.piezo_scan_pos[0])
        piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[1])
        piezo_sequences[1] = np.repeat(self.piezo_scan_positions[1], digital_sequences[0].shape[0])
        piezo_sequences[1] = shift_array(piezo_sequences[1], idle_samples, piezo_sequences[1][0], "backward")
        for i, dtr in enumerate(digital_sequences):
            digital_sequences[i] = np.tile(dtr, self.piezo_scan_pos[1])
        for i, gtr in enumerate(galvo_sequences):
            galvo_sequences[i] = np.tile(gtr, self.piezo_scan_pos[1])
        scan_pos = self.piezo_scan_pos[0] * self.piezo_scan_pos[1]
        self.logg.info("\nGalvo start, and stop: {} \n"
                       "Dot start, step, range, and numbers: {} \n"
                       "Piezo starts: {} \n"
                       "Piezo steps: {} \n"
                       "Piezo ranges: {} \n"
                       "Piezo positions: {}".format([self.galvo_starts, self.galvo_stops],
                                                    [self.dot_starts, [self.dot_step_v, self.dot_step_y],
                                                     self.dot_ranges, self.dot_pos.size], self.piezo_starts,
                                                    self.piezo_steps, self.piezo_ranges, scan_pos))
        return np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences), lasers, scan_pos


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
