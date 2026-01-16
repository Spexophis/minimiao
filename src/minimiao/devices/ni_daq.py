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
from nidaqmx.stream_readers import CounterReader
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
        self.mode = None
        self.galvo_channels = ["Dev1/ao0", "Dev1/ao1"]
        self.piezo_channels = ["Dev1/ao2"]
        self.ttl_channels = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line3",
                                 "Dev1/port0/line4", "Dev1/port0/line5"]
        self.photon_counter_channels = ["/Dev1/ctr0", "/Dev1/ctr1"]
        self.photon_counter_terminals = ["/Dev1/PFI0", "/Dev1/PFI12"]
        self.pmt_channel = ["/Dev1/ai0"]
        self._photon_counter_length = int(2 ** 16)
        self.photon_reader = None
        self.pmt_reader = None
        self.photon_counter_mode = 0
        self.psr = None

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
            tasks = {"analog": None, "digital": None, "photon_counter": None, "pmt_reader": None}
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

    @property
    def photon_counter_length(self) -> int:
        return self._photon_counter_length

    @photon_counter_length.setter
    def photon_counter_length(self, value: int) -> None:
        self._photon_counter_length = max(int(value), int(2 ** 16))

    def set_galvo_position(self, pos, indices=None):
        if indices is None:
            indices = [0, 1]
        if len(pos) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping galvo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.galvo_channels[ind], min_val=-10., max_val=10.)
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

    def get_galvo_position(self):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai0:1", min_val=-10.0, max_val=10.0)
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

    def set_piezo_position(self, pos, indices=None):
        if indices is None:
            indices = [0]
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
                task.ai_channels.add_ai_voltage_chan("Dev1/ai2", min_val=-10.0, max_val=10.0)
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

    def write_analog_sequences(self, analog_sequences, analog_channels):
        if analog_sequences.ndim > 1:
            n_channels, n_samples = analog_sequences.shape
        else:
            n_channels = 1
            n_samples = analog_sequences.shape[0]
        if n_channels != len(analog_channels):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping piezo sequences update.")
            return
        try:
            self.tasks["analog"] = nidaqmx.Task("analog")
            for analog_channel in analog_channels:
                self.tasks["analog"].ao_channels.add_ao_voltage_chan(analog_channel, min_val=-10., max_val=10.)
            self.tasks["analog"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                            sample_mode=self.mode,
                                                            samps_per_chan=n_samples)
            self.tasks["analog"].write(analog_sequences, auto_start=False)
            self._active["analog"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_digital_sequences(self, digital_sequences, digital_channels):
        if digital_sequences.ndim > 1:
            n_channels, n_samples = digital_sequences.shape
            if n_channels == 1:
                digital_sequences = digital_sequences[0]
        else:
            n_channels = 1
            n_samples = digital_sequences.shape[0]
        if n_channels != len(digital_channels):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping digital sequences update.")
            return
        try:
            self.tasks["digital"] = nidaqmx.Task("digital")
            for digital_channel in digital_channels:
                self.tasks["digital"].do_channels.add_do_chan(digital_channel,
                                                              line_grouping=LineGrouping.CHAN_PER_LINE)
            self.tasks["digital"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                             source="/Dev1/ao/SampleClock",
                                                             sample_mode=self.mode,
                                                             samps_per_chan=n_samples)
            self.tasks["digital"].triggers.start_trigger.cfg_dig_edge_start_trig("/Dev1/ao/StartTrigger")
            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)
            self._active["digital"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self,
                       analog_sequences=None, analog_channels=None,
                       digital_sequences=None, digital_channels=None,
                       finite=True):
        if finite:
            self.mode = AcquisitionType.FINITE
        else:
            self.mode = AcquisitionType.CONTINUOUS
        try:
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences, digital_channels)
            if analog_sequences is not None:
                self.write_analog_sequences(analog_sequences, analog_channels)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def prepare_photon_counter(self):
        self.tasks["photon_counter"] = nidaqmx.Task("photon_counter")
        c0 = self.tasks["photon_counter"].ci_channels.add_ci_count_edges_chan(counter=self.photon_counter_channels[0],
                                                                              edge=Edge.RISING)
        c0.ci_count_edges_term = self.photon_counter_terminals[0]
        c1 = self.tasks["photon_counter"].ci_channels.add_ci_count_edges_chan(counter=self.photon_counter_channels[1],
                                                                              edge=Edge.RISING)
        c1.ci_count_edges_term = self.photon_counter_terminals[1]
        self.tasks["photon_counter"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source="/Dev1/ao/SampleClock",
                                                                active_edge=Edge.RISING, sample_mode=self.mode,
                                                                samps_per_chan=self.photon_counter_length)
        self.tasks["photon_counter"].triggers.start_trigger.cfg_dig_edge_start_trig("/Dev1/ao/StartTrigger")
        self.tasks["photon_counter"].in_stream.input_buf_size = self.photon_counter_length
        self.photon_reader = CounterReader(self.tasks["photon_counter"].in_stream)
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
                counts = np.empty((2, avail), dtype=np.uint32)
                self.photon_reader.read_many_sample_uint32(data=counts, number_of_samples_per_channel=avail, timeout=0.0)
                # counts = self.tasks["photon_counter"].read(number_of_samples_per_channel=avail, timeout=0.0)
                self.data.add_element(counts, avail)
        except nidaqmx.DaqWarning as e:
            self.logg.error("DAQ read error %s: %s", e.error_code, e)

    def get_data(self):
        edg_num, count_data = self.data.get_elements()
        return count_data

    def prepare_pmt_reader(self):
        self.tasks["pmt_reader"] = nidaqmx.Task("pmt_reader")
        self.tasks["pmt_reader"].ai_channels.add_ai_voltage_chan(self.pmt_channel, min_val=-10., max_val=10.)
        self.tasks["pmt_reader"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source="/Dev1/ao/SampleClock",
                                                     sample_mode=self.mode, samps_per_chan=self.photon_counter_length)
        self.tasks["pmt_reader"].in_stream.input_buf_size = self.photon_counter_length
        self.pmt_reader = AnalogSingleChannelReader(self.tasks["pmt_reader"].in_stream)
        self._active["pmt_reader"] = True
        
    def get_pmt(self):
        try:
            avail = self.tasks["pmt_reader"].in_stream.avail_samp_per_chan
            if avail > 0:
                amps = np.empty(avail, dtype=np.float64)
                self.pmt_reader.read_many_sample(data=amps, number_of_samples_per_channel=avail, timeout=0.0)
                # amps = self.tasks["pmt_reader"].read(number_of_samples_per_channel=avail, timeout=0.0)
                # self.data.add_element(counts, avail)
        except nidaqmx.DaqWarning as e:
            self.logg.error("DAQ read error %s: %s", e.error_code, e)

    def start_triggers(self):
        try:
            if self._active["digital"]:
                self.tasks["digital"].start()
                self._running["digital"] = True
            if self._active["photon_counter"]:
                self.tasks["photon_counter"].start()
                self._running["photon_counter"] = True
            if self._active["analog"]:
                self.tasks["analog"].start()
                self._running["analog"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)

    def run_triggers(self):
        try:
            self.start_triggers()
            if self.mode == AcquisitionType.FINITE:
                for key, _task in self.tasks.items():
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
