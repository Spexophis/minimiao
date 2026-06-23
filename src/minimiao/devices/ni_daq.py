# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import warnings

import nidaqmx
import numpy as np
from nidaqmx.constants import Signal, Edge, AcquisitionType, LineGrouping, WAIT_INFINITELY, \
    DataTransferActiveTransferMode, READ_ALL_AVAILABLE
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
from nidaqmx.system import System

from minimiao import run_threads, logger

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.devices = self._initialize()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()
        self.mpd_data = None
        self.acq_threads = []
        self.sample_rate = 50e3
        self.duty_cycle = float(0.5)
        self.mode = None
        self.analog_output_channels = ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2"]
        self.analog_input_channels = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2"]
        self.analog_reader = None
        self.analog_data = None
        self.galvo_channels = ["Dev1/ao0", "Dev1/ao1"]
        self.piezo_channels = ["Dev1/ao2"]
        self.digital_output_channels = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line2",
                                        "Dev1/port0/line3", "Dev1/port0/line4", "Dev1/port0/line5"]
        self.photon_counter_channels = ["/Dev1/ctr0", "/Dev1/ctr1"]
        self.photon_counter_terminals = ["/Dev1/PFI0", "/Dev1/PFI12", "/Dev1/PFI3"]
        self.pmt_data = None
        self.pmt_thread = None
        self._photon_counter_length = int(2 ** 16)

    def close(self):
        for device in self.devices:
            device.reset_device()

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
            tasks = {"analog_out": None, "digital_out": None, "analog_in": None, "photon_counters": []}
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
        self._photon_counter_length = int(value)

    def set_galvo_position(self, pos, indices=None):
        if indices is None:
            indices = [0, 1]
        if len(pos) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping galvo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.analog_output_channels[ind], min_val=-10., max_val=10.)
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
        indices = [0, 1]
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ai_channels.add_ai_voltage_chan(self.analog_input_channels[ind], min_val=-10.0, max_val=10.0)
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
            indices = [2]
        if len(pos) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping piezo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.analog_output_channels[ind], min_val=0., max_val=10.)
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

    def get_piezo_position(self, indices=None):
        if indices is None:
            indices = [2]
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ai_channels.add_ai_voltage_chan(self.analog_input_channels[ind], min_val=-1.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=self.sample_rate, sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=16, active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=16)
            return sum(pos) / len(pos)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_analog_sequences(self, analog_sequences, analog_channels, reader_channels=None):
        if analog_sequences.ndim > 1:
            n_channels, n_samples = analog_sequences.shape
        else:
            n_channels = 1
            n_samples = analog_sequences.shape[0]
        if n_channels != len(analog_channels):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping piezo sequences update.")
            return
        try:
            self.tasks["analog_out"] = nidaqmx.Task("analog_out")
            for analog_channel in analog_channels:
                self.tasks["analog_out"].ao_channels.add_ao_voltage_chan(self.analog_output_channels[analog_channel],
                                                                         min_val=-10., max_val=10.)
            self.tasks["analog_out"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                                sample_mode=self.mode, samps_per_chan=n_samples)
            self.tasks["analog_out"].out_stream.regen_mode = nidaqmx.constants.RegenerationMode.ALLOW_REGENERATION
            self.tasks["analog_out"].export_signals.export_signal(Signal.SAMPLE_CLOCK, "/Dev1/PFI1")
            self.tasks["analog_out"].write(analog_sequences, auto_start=False)
            self._active["analog_out"] = True
            if reader_channels is not None:
                self.tasks["analog_in"] = nidaqmx.Task("analog_in")
                self.analog_data = np.zeros((len(reader_channels), n_samples))
                for reader_channel in reader_channels:
                    self.tasks["analog_in"].ai_channels.add_ai_voltage_chan(self.analog_input_channels[reader_channel],
                                                                            min_val=-10., max_val=10.)
                self.tasks["analog_in"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source="/Dev1/PFI2",
                                                                   sample_mode=self.mode, samps_per_chan=n_samples)
                if len(reader_channels) > 1:
                    self.analog_reader = AnalogMultiChannelReader(self.tasks["analog_in"].in_stream)
                else:
                    self.analog_reader = AnalogSingleChannelReader(self.tasks["analog_in"].in_stream)
                if 0 in reader_channels:
                    self.pmt_data = run_threads.PMTList(n_samples)
                    if self.mode == nidaqmx.constants.AcquisitionType.CONTINUOUS:
                        self.pmt_thread = run_threads.PMTThread(self)
                self._active["analog_in"] = True
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
            self.tasks["digital_out"] = nidaqmx.Task("digital_out")
            for digital_channel in digital_channels:
                self.tasks["digital_out"].do_channels.add_do_chan(self.digital_output_channels[digital_channel],
                                                                  line_grouping=LineGrouping.CHAN_PER_LINE)
            self.tasks["digital_out"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source="/Dev1/PFI2",
                                                                 sample_mode=self.mode, samps_per_chan=n_samples)
            self.tasks["digital_out"].timing.samp_clk_active_edge = Edge.RISING
            self.tasks["digital_out"].write(digital_sequences == 1.0, auto_start=False)
            self._active["digital_out"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self, analog_sequences=None, analog_channels=None,
                       digital_sequences=None, digital_channels=None,
                       reader_channels=None, finite=True):
        if finite:
            self.mode = AcquisitionType.FINITE
        else:
            self.mode = AcquisitionType.CONTINUOUS
        try:
            if analog_sequences is not None:
                self.write_analog_sequences(analog_sequences, analog_channels, reader_channels)
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences, digital_channels)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def prepare_photon_counter(self, cl: list):
        self.tasks["photon_counters"] = []
        self.acq_threads = []
        for i in range(len(cl)):
            tsk_name = f"photon_counter_{i}"
            self.tasks["photon_counters"].append(nidaqmx.Task(tsk_name))
            self.acq_threads.append(run_threads.PhotonCountThread(self, i))
        self.mpd_data = run_threads.PhotonCountList(len(cl), self.photon_counter_length)
        for n, tsk in enumerate(self.tasks["photon_counters"]):
            c = tsk.ci_channels.add_ci_count_edges_chan(counter=self.photon_counter_channels[n],
                                                        edge=Edge.RISING)
            c.ci_count_edges_term = self.photon_counter_terminals[cl[n]]
            c.ci_data_xfer_mech = DataTransferActiveTransferMode.DMA
            tsk.timing.cfg_samp_clk_timing(rate=self.sample_rate, source="/Dev1/PFI2",
                                           active_edge=Edge.RISING, sample_mode=self.mode,
                                           samps_per_chan=self.photon_counter_length)
            tsk.in_stream.input_buf_size = self.photon_counter_length
        self._active["photon_counters"] = True

    def clear_photon_counter(self):
        self.acq_threads = []
        self.mpd_data = None
        self._active["photon_counters"] = False

    def start_photon_count(self):
        for acq in self.acq_threads:
            if acq:
                if acq.started:
                    acq.resume()
                    self.logg.info("Photon counting resumed")
                else:
                    acq.start()
                    self.logg.info("Photon counting started")

    def pause_photon_count(self):
        for acq in self.acq_threads:
            if acq:
                acq.pause()
        self.logg.info("Photon counting paused")

    def resume_photon_count(self):
        for acq in self.acq_threads:
            if acq:
                acq.resume()
        self.logg.info("Photon counting resumed")

    def stop_photon_count(self):
        for acq in self.acq_threads:
            if acq:
                acq.stop()
                acq = None
        self.logg.info("Photon counting stopped")

    def get_photon_counts(self, ind):
        try:
            avail = self.tasks["photon_counters"][ind].in_stream.avail_samp_per_chan
            # total = self.tasks["photon_counters"][ind].in_stream.total_samp_per_chan_acquired
            if avail > 0:
                counts = self.tasks["photon_counters"][ind].read(number_of_samples_per_channel=avail, timeout=0.0)
                self.mpd_data.add_element(counts, avail, ind)
        except nidaqmx.DaqWarning as e:
            self.logg.error("DAQ read error %s: %s", e.error_code, e)

    def get_data(self):
        edg_num, count_data = self.mpd_data.get_elements()
        return count_data

    def start_pmt_read(self):
        if self.pmt_thread:
            if self.pmt_thread.started:
                self.pmt_thread.resume()
                self.logg.info("PMT reading resumed")
            else:
                self.pmt_thread.start()
                self.logg.info("PMT reading started")

    def stop_pmt_read(self):
        if self.pmt_thread:
            self.pmt_thread.stop()
            self.pmt_thread = None
        self.logg.info("PMT reading stopped")

    def get_pmt_voltages(self):
        try:
            avail = self.tasks["analog_in"].in_stream.avail_samp_per_chan
            if avail > 0:
                if isinstance(self.analog_reader, AnalogMultiChannelReader):
                    tmp = np.empty((self.analog_data.shape[0], avail), dtype=np.float64)
                    self.analog_reader.read_many_sample(tmp, number_of_samples_per_channel=avail, timeout=0.)
                    pmt_ch = - tmp[0]
                else:
                    pmt_ch = np.empty(avail, dtype=np.float64)
                    self.analog_reader.read_many_sample(pmt_ch, number_of_samples_per_channel=avail, timeout=0.)
                    pmt_ch = - pmt_ch
                self.pmt_data.add_element(pmt_ch.tolist(), avail)
        except nidaqmx.DaqWarning as e:
            self.logg.error("PMT read error %s: %s", e.error_code, e)

    def start_triggers(self):
        try:
            if self._active["digital_out"]:
                self.tasks["digital_out"].start()
                self._running["digital_out"] = True
            if self._active["photon_counters"]:
                for task in self.tasks["photon_counters"]:
                    task.start()
                self._running["photon_counters"] = True
                self.start_photon_count()
            if self._active["analog_in"]:
                self.tasks["analog_in"].start()
                self._running["analog_in"] = True
                if self.pmt_thread is not None:
                    self.start_pmt_read()
            if self._active["analog_out"]:
                self.tasks["analog_out"].start()
                self._running["analog_out"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)

    def run_triggers(self):
        try:
            self.start_triggers()
            if self.mode == AcquisitionType.FINITE:
                try:
                    self.logg.info("Trigger is running")
                    if self._active["analog_out"] and self._running["analog_out"]:
                        self.tasks["analog_out"].wait_until_done(WAIT_INFINITELY)
                    if self._active["analog_in"] and self._running["analog_in"]:
                        self.tasks["analog_in"].wait_until_done(WAIT_INFINITELY)
                        self.analog_reader.read_many_sample(data=self.analog_data,
                                                            number_of_samples_per_channel=READ_ALL_AVAILABLE)
                        if self.pmt_data is not None:
                            pmt_ch = -self.analog_data[0]
                            self.pmt_data.add_element(pmt_ch.tolist(), pmt_ch.shape[0])
                    if self._active["photon_counters"] and self._running["photon_counters"]:
                        for task in self.tasks["photon_counters"]:
                            task.wait_until_done(WAIT_INFINITELY)
                    if self._active["digital_out"] and self._running["digital_out"]:
                        self.tasks["digital_out"].wait_until_done(WAIT_INFINITELY)
                    self.logg.info("Trigger finished")
                except nidaqmx.DaqWarning as e:
                    self.logg.warning("DaqWarning caught as exception: %s", e)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def stop_triggers(self, _close=True):
        try:
            if self._active["analog_out"] and self._running["analog_out"]:
                self.tasks["analog_out"].stop()
                self._running["analog_out"] = False
            if self._active["analog_in"] and self._running["analog_in"]:
                self.tasks["analog_in"].stop()
                self._running["analog_in"] = False
            if self._active["photon_counters"] and self._running["photon_counters"]:
                for task in self.tasks["photon_counters"]:
                    task.stop()
                self._running["photon_counters"] = False
            if self._active["digital_out"] and self._running["digital_out"]:
                self.tasks["digital_out"].stop()
                self._running["digital_out"] = False
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
        if _close:
            self.close_triggers()

    def close_triggers(self):
        try:
            if self._active["analog_out"]:
                self.tasks["analog_out"].close()
                self.tasks["analog_out"] = None
                self._active["analog_out"] = False
            if self._active["analog_in"]:
                self.tasks["analog_in"].close()
                self.tasks["analog_in"] = None
                self._active["analog_in"] = False
                self.analog_reader = None
            if self._active["photon_counters"]:
                for task in self.tasks["photon_counters"]:
                    task.close()
                    task = None
                self._active["photon_counters"] = False
            if self._active["digital_out"]:
                self.tasks["digital_out"].close()
                self.tasks["digital_out"] = None
                self._active["digital_out"] = False
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
