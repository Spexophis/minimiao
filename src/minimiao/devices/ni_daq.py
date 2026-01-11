# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import warnings

import nidaqmx
import numpy as np
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, WAIT_INFINITELY
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogMultiChannelWriter
from nidaqmx.system import System

from minimiao import run_threads

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.setup_logging()
        self.devices = self._initialize()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()
        self.data = None
        self.acq_thread = None
        self.sample_rate = int(250000)
        self.duty_cycle = float(0.5)
        self.piezo_channels = ["Dev3/ao0", "Dev3/ao1", "Dev3/ao2"]
        self.digital_channels = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line3",
                                 "Dev1/port0/line4", "Dev1/port0/line5", "Dev1/port0/line6"]
        self.photon_counter_channel = "/Dev1/ctr1"
        self.photon_counter_terminal = "/Dev1/PFI0"
        self._photon_counter_length = int(2 ** 16)
        self.photon_counter_mode = 0
        self.psr = None
        self.clock_counter_channel = "/Dev1/ctr0"
        self.clock_counter_terminals = ["/Dev1/PFI12", "/Dev3/PFI0"]
        self.mode = None

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
            tasks = {"piezo": None, "digital": None, "photon_counter": None, "clock": None, "gate": None}
            _active = {key: False for key in tasks.keys()}
            _running = {key: False for key in tasks.keys()}
            return tasks, _active, _running
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
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
                    task.ao_channels.add_ao_voltage_chan(self.piezo_channels[ind], min_val=0., max_val=10.)
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

    def write_clock_channel(self):
        try:
            self.tasks["clock"] = nidaqmx.Task("clock")
            co_channel = self.tasks["clock"].co_channels.add_co_pulse_chan_freq(counter=self.clock_counter_channel,
                                                                                freq=self.sample_rate,
                                                                                duty_cycle=self.duty_cycle)
            co_channel.co_ctr_timebase_src = '20MHzTimebase'
            co_channel.co_pulse_term = self.clock_counter_terminals[0]
            self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            self._active["clock"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_digital_sequences(self, digital_sequences, indices=None, callback=None):
        if indices is None:
            indices = [0, 1, 2, 3, 4]
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
                                                             active_edge=Edge.RISING, sample_mode=self.mode,
                                                             samps_per_chan=n_samples)
            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)
            self._active["digital"] = True
            if callback is not None:
                self.tasks["digital"].register_signal_event(nidaqmx.constants.Signal.CHANGE_DETECTION_EVENT, callback)
                self.tasks["digital"].change_detection.dig_edge_start_trig.cfg_dig_edge_start_trig(
                    trigger_source=self.digital_channels[4], trigger_edge=nidaqmx.constants.Edge.FALLING)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_piezo_sequences(self, piezo_sequences, indices=None):
        if indices is None:
            indices = [0, 1, 2]
        if piezo_sequences.ndim > 1:
            n_channels, n_samples = piezo_sequences.shape
        else:
            n_channels = 1
            n_samples = piezo_sequences.shape[0]
        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping piezo sequences update.")
            return
        try:
            self.tasks["piezo"] = nidaqmx.Task("piezo")
            for ind in indices:
                self.tasks["piezo"].ao_channels.add_ao_voltage_chan(self.piezo_channels[ind], min_val=0., max_val=10.)
            self.tasks["piezo"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                           source=self.clock_counter_terminals[1],
                                                           active_edge=Edge.RISING, sample_mode=self.mode,
                                                           samps_per_chan=n_samples)
            self.tasks["piezo"].write(piezo_sequences, auto_start=False)
            self._active["piezo"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self, piezo_sequences=None, piezo_channels=None,
                       digital_sequences=None, digital_channels=None,
                       finite=True):
        self.write_clock_channel()
        if finite:
            self.mode = AcquisitionType.FINITE
        else:
            self.mode = AcquisitionType.CONTINUOUS
        try:
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences, indices=digital_channels)
            if piezo_sequences is not None:
                self.write_piezo_sequences(piezo_sequences, indices=piezo_channels)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    @property
    def photon_counter_length(self) -> int:
        return self._photon_counter_length

    @photon_counter_length.setter
    def photon_counter_length(self, value: int) -> None:
        self._photon_counter_length = max(int(value), int(2 ** 16))

    def prepare_photon_counter(self):
        if self.tasks["clock"] is None:
            self.write_clock_channel()

        self.tasks["photon_counter"] = nidaqmx.Task("photon_counter")
        ci = self.tasks["photon_counter"].ci_channels.add_ci_count_edges_chan(counter=self.photon_counter_channel, edge=Edge.RISING)
        ci.ci_count_edges_term = self.photon_counter_terminal
        self.tasks["photon_counter"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock_counter_terminals[0],
                                     active_edge=Edge.RISING, sample_mode=self.mode,
                                     samps_per_chan=self.photon_counter_length)
        self.tasks["photon_counter"].in_stream.input_buf_size = self.photon_counter_length
        self.data = run_threads.PhotonCountList(self.photon_counter_length)
        self.acq_thread = run_threads.PhotonCountThread(self)
        if self.photon_counter_mode:
            self.data.on_update(self.psr.point_scan_live_recon)
        self._active["photon_counter"] = True

    def start_photon_count(self):
        self.acq_thread.start()
        self.logg.info("Photon counting started")

    def stop_photon_count(self):
        if self.acq_thread:
            self.acq_thread.stop()
            self.acq_thread = None
        self.logg.info("Photon counting stopped")

    def get_photon_count(self):
        try:
            avail = self.tasks["photon_counter"].in_stream.avail_samp_per_chan
            if avail > 0:
                counts = self.tasks["photon_counter"].read(number_of_samples_per_channel=avail, timeout=0.0)
                self.data.add_element(counts, avail)
        except nidaqmx.DaqWarning as e:
            self.logg.error("DAQ read error %s: %s", e.error_code, e)

    def get_data(self):
        edg_num, count_data = self.data.get_elements()
        return count_data

    def start_triggers(self):
        try:
            for key, _task in self.tasks.items():
                if key != "clock":
                    if self._active.get(key, False):
                        if not self._running[key]:
                            _task.start()
                            self._running[key] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)

    def run_triggers(self):
        try:
            self.start_triggers()
            self._running["clock"] = True
            self.tasks["clock"].start()
            if self.mode == AcquisitionType.FINITE:
                for key, _task in self.tasks.items():
                    if key != "clock":
                        if self._active.get(key, False):
                            if self._running.get(key, False):
                                _task.wait_until_done(WAIT_INFINITELY)
            self.logg.info("Trigger is running")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def stop_triggers(self, _close=True):
        for key, _task in self.tasks.items():
            if self._active.get(key, False):
                if self._running.get(key, False):
                    _task.stop()
        self._running = {key: False for key in self._running}
        if _close:
            self.close_triggers()

    def close_triggers(self):
        for key, _task in self.tasks.items():
            if self._active.get(key, False):
                _task.close()
                _task = None
        self._active = {key: False for key in self._active}

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
