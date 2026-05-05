# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import warnings

import nidaqmx
import numpy as np
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, WAIT_INFINITELY, TaskMode, RegenerationMode
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogMultiChannelWriter
from nidaqmx.system import System

from minimiao import run_threads, logger

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.setup_logging()
        self.devices = self._initialize()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()
        self.sample_rate = int(200000)
        self.duty_cycle = float(0.5)
        self.analog_channels = ["Dev3/ao0", "Dev3/ao1", "Dev3/ao2"]
        self.digital_channels = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line3",
                                 "Dev1/port0/line4", "Dev1/port0/line5", "Dev1/port0/line6"]
        self.clock_external_start_terminal = "/Dev1/PFI1"
        self.clock_counter_channel = "/Dev1/ctr0"
        self.clock_counter_terminals = ["/Dev1/PFI12", "/Dev3/PFI0"]
        self.run_mode = None
        self.clock_triggered = False
        self.sequence_samples = None

    def __del__(self):
        pass

    def close(self):
        for device in self.devices:
            device.reset_device()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _initialize(self):
        try:
            local_system = System.local()
            driver_version = local_system.driver_version
            self.logg.info("DAQmx {0}.{1}.{2}".format(driver_version.major_version, driver_version.minor_version,
                                                      driver_version.update_version))
            return local_system.devices
        except Exception as e:
            self.logg.error(f"Error initializing NIDAQ: {e}")

    def _configure(self):
        try:
            tasks = {"digital": None, "analog": None, "clock": None}
            _active = {key: False for key in tasks.keys()}
            _running = {key: False for key in tasks.keys()}
            return tasks, _active, _running

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, \
                    "Unexpected error code: {}".format(e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_piezo_position(self, pos, indices=None):
        if indices is None:
            indices = [0, 1, 2]
        if len(pos) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping piezo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.analog_channels[ind], min_val=0., max_val=10.)
                task.write(pos)
                task.wait_until_done(WAIT_INFINITELY)
                task.stop()
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def get_piezo_position(self):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev3/ai0:2", min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=self.sample_rate, sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=16, active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=16)
            return [sum(p) / len(p) for p in pos]
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_clock_channel(self, samples_per_trigger=0, trigger=False):
        try:
            self.tasks["clock"] = nidaqmx.Task("clock")
            co_channel = self.tasks["clock"].co_channels.add_co_pulse_chan_freq(counter=self.clock_counter_channel,
                                                                                freq=self.sample_rate,
                                                                                duty_cycle=self.duty_cycle)
            co_channel.co_ctr_timebase_src = '20MHzTimebase'
            co_channel.co_pulse_term = self.clock_counter_terminals[0]
            if samples_per_trigger > 0:
                self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE,
                                                               samps_per_chan=samples_per_trigger)
            else:
                self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            if trigger:
                self.tasks["clock"].triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=self.clock_external_start_terminal,
                                                                                   trigger_edge=Edge.RISING)
                self.tasks["clock"].triggers.start_trigger.retriggerable = True
            self.tasks["clock"].control(TaskMode.TASK_COMMIT)
            self._active["clock"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_digital_sequences(self, digital_sequences, indices=None):
        if indices is None:
            indices = [0, 1, 2, 3, 4]

        digital_sequences = np.asarray(digital_sequences)

        if digital_sequences.ndim > 1:
            n_channels, n_samples = digital_sequences.shape
            if n_channels == 1:
                digital_sequences = digital_sequences[0]
        else:
            n_channels = 1
            n_samples = digital_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping digital sequences update.")
            return

        try:
            self.tasks["digital"] = nidaqmx.Task("digital")

            for ind in indices:
                self.tasks["digital"].do_channels.add_do_chan(self.digital_channels[ind],
                                                              line_grouping=LineGrouping.CHAN_PER_LINE)
            self.tasks["digital"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                             source=self.clock_counter_terminals[0],
                                                             active_edge=Edge.RISING, sample_mode=self.run_mode,
                                                             samps_per_chan=n_samples)
            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.tasks["digital"].out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)
            self._active["digital"] = True

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_analog_sequences(self, piezo_sequences, indices=None):
        if indices is None:
            indices = [0, 1, 2]

        piezo_sequences = np.asarray(piezo_sequences)

        if piezo_sequences.ndim > 1:
            n_channels, n_samples = piezo_sequences.shape
        else:
            n_channels = 1
            n_samples = piezo_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping piezo sequences update.")
            return

        try:
            self.tasks["analog"] = nidaqmx.Task("analog")
            for ind in indices:
                self.tasks["analog"].ao_channels.add_ao_voltage_chan(self.analog_channels[ind], min_val=0., max_val=10.)
            self.tasks["analog"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                           source=self.clock_counter_terminals[1],
                                                           active_edge=Edge.RISING, sample_mode=self.run_mode,
                                                           samps_per_chan=n_samples)
            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.tasks["analog"].out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            self.tasks["analog"].write(piezo_sequences, auto_start=False)
            self._active["analog"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self, digital_sequences=None, digital_channels=None,
                       analog_sequences=None, analog_channels=None,
                       finite=True, trg=False):
        sequence_samples = None

        if analog_sequences is not None:
            analog_sequences = np.asarray(analog_sequences)

            if analog_sequences.ndim > 1:
                n_analog_channel, analog_samples = analog_sequences.shape
            else:
                n_analog_channel = 1
                analog_samples = analog_sequences.shape[0]

            if analog_channels is None:
                analog_channels = list(range(n_analog_channel))

            if n_analog_channel != len(analog_channels):
                self.logg.error(
                    "WARNING: Length of n_analog_channel and analog_channels differ, "
                    "skipping piezo sequences."
                )
                return

            sequence_samples = analog_samples

        if digital_sequences is not None:
            digital_sequences = np.asarray(digital_sequences)

            if digital_sequences.ndim > 1:
                n_digital_channel, digital_samples = digital_sequences.shape
            else:
                n_digital_channel = 1
                digital_samples = digital_sequences.shape[0]

            if digital_channels is None:
                digital_channels = list(range(n_digital_channel))

            if n_digital_channel != len(digital_channels):
                self.logg.error(
                    "WARNING: Length of n_digital_channel and digital_channels differ, "
                    "skipping digital sequences."
                )
                return

            if sequence_samples is None:
                sequence_samples = digital_samples
            elif sequence_samples != digital_samples:
                self.logg.error(
                    "WARNING: Length of digital sequences and analog sequences differ."
                )
                return

        if sequence_samples is None:
            self.logg.error("No analog or digital sequence was provided.")
            return

        self.sequence_samples = sequence_samples

        if finite:
            self.run_mode = AcquisitionType.FINITE
            clock_samples = sequence_samples
        else:
            self.run_mode = AcquisitionType.CONTINUOUS
            if trg:
                clock_samples = sequence_samples
            else:
                clock_samples = 0
        self.write_clock_channel(samples_per_trigger=clock_samples, trigger=trg)

        try:
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences, indices=digital_channels)
            if analog_sequences is not None:
                self.write_analog_sequences(analog_sequences, indices=analog_channels)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def start_triggers(self):
        try:
            for key, _task in self.tasks.items():
                if key == "clock":
                    continue

                if _task is None:
                    continue

                if self._active.get(key, False):
                    if not self._running.get(key, False):
                        _task.start()
                        self._running[key] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)

    def run_triggers(self):
        try:
            if self.tasks["clock"] is None:
                self.logg.error("Clock task has not been configured.")
                return

            self.start_triggers()

            if not self._running.get("clock", False):
                self.tasks["clock"].start()
                self._running["clock"] = True

            self.logg.info("Trigger is running")

            if self.run_mode == AcquisitionType.FINITE:
                for key, _task in self.tasks.items():
                    if key == "clock":
                        continue

                    if _task is None:
                        continue

                    if self._active.get(key, False):
                        _task.wait_until_done(WAIT_INFINITELY)
                        self._running[key] = False

                self.logg.info("Finite trigger sequence finished.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def stop_triggers(self, _close=True):
        for key, _task in self.tasks.items():
            if _task is None:
                continue

            if self._active.get(key, False):
                if self._running.get(key, False):
                    try:
                        _task.stop()
                    except nidaqmx.DaqWarning as e:
                        self.logg.warning("DaqWarning caught as exception: %s", e)

        self._running = {key: False for key in self._running}

        if _close:
            self.close_triggers()

    def close_triggers(self):
        for key, _task in self.tasks.items():
            if _task is None:
                continue

            if self._active.get(key, False):
                try:
                    _task.close()
                except nidaqmx.DaqWarning as e:
                    self.logg.warning("DaqWarning caught as exception: %s", e)

                self.tasks[key] = None

        self._active = {key: False for key in self._active}
        self._running = {key: False for key in self._running}

    def measure_ao(self, output_channels, input_channels, data):
        if data.ndim > 1:
            _, num_samples = data.shape
        else:
            num_samples = data.shape[0]
        acquired_data = np.zeros(data.shape)
        with nidaqmx.Task() as clk_task:
            co_channel = clk_task.co_channels.add_co_pulse_chan_freq(counter=self.clock_counter_channel,
                                                                     freq=self.sample_rate, duty_cycle=self.duty_cycle)
            co_channel.co_pulse_term = self.clock_counter_terminals[0]
            clk_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            with nidaqmx.Task() as output_task:
                output_task.ao_channels.add_ao_voltage_chan(output_channels, min_val=-10., max_val=10.)
                output_task.timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock_counter_terminals[1],
                                                       active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE,
                                                       samps_per_chan=num_samples)
                with nidaqmx.Task() as input_task:
                    input_task.ai_channels.add_ai_voltage_chan(input_channels, min_val=-10., max_val=10.)
                    input_task.timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock_counter_terminals[1],
                                                          sample_mode=AcquisitionType.FINITE,
                                                          samps_per_chan=num_samples)
                    if data.ndim > 1:
                        writer = AnalogMultiChannelWriter(output_task.out_stream)
                        reader = AnalogMultiChannelReader(input_task.in_stream)
                    else:
                        writer = AnalogSingleChannelWriter(output_task.out_stream)
                        reader = AnalogSingleChannelReader(input_task.in_stream)
                    writer.write_many_sample(data)
                    input_task.start()
                    output_task.start()
                    clk_task.start()
                    output_task.wait_until_done(WAIT_INFINITELY)
                    input_task.wait_until_done(WAIT_INFINITELY)
                    reader.read_many_sample(data=acquired_data, number_of_samples_per_channel=num_samples)
        return acquired_data
