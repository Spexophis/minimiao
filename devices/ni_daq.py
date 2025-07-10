import warnings

import nidaqmx
import numpy as np
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, FrequencyUnits, Level, WAIT_INFINITELY
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.stream_readers import AnalogSingleChannelReader  # , AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter  # , AnalogMultiChannelWriter
from nidaqmx.system import System

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:
    class NIDAQSettings:

        def __init__(self):
            self.sample_rate = 250000
            self.duty_cycle = 0.5
            self.galvo_channels = ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2"]
            self.piezo_channels = ["Dev2/ao0", "Dev2/ao1", "Dev2/ao2"]
            self.digital_channels = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line2", "Dev1/port0/line3",
                                     "Dev1/port0/line4", "Dev1/port0/line5", "Dev1/port0/line6"]
            self.counter_channel = "/Dev1/ctr0"
            self.clock_rate = 2000000
            self.clock = ["/Dev1/PFI12", "/Dev2/PFI0"]
            self.mode = None

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.setup_logging()
        self.devices = self._initialize()
        self._settings = self.NIDAQSettings()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()

    def __del__(self):
        pass

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

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
            tasks = {"piezo": None, "galvo": None, "switch": None, "digital": None, "clock": None}
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
                task.ai_channels.add_ai_voltage_chan("Dev2/ai0:2", min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=200000, sample_mode=AcquisitionType.FINITE, samps_per_chan=10,
                                                active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=10)
            return [sum(p) / len(p) for p in pos]
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def piezo_step_callback(self, task_handle, signal_type, callback_data):
        self.tasks["piezo_x"].write(self.ao_value[self.pzx], auto_start=True)
        self.pzx += 1
        if self.pzx >= 20:
            self.pzy += 1
            self.tasks["piezo_y"].write(self.ao_value[self.pzy], auto_start=True)
            self.pzx = 0
            self.tasks["piezo_x"].write(self.ao_value[self.pzx], auto_start=True)
        return 0

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
                task.timing.cfg_samp_clk_timing(rate=500000, sample_mode=AcquisitionType.FINITE, samps_per_chan=10,
                                                active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=10)
            return [sum(p) / len(p) for p in pos]
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_switch_position(self, pos):
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.galvo_channels[2], min_val=-5., max_val=5.)
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

    def get_switch_position(self):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai2", min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=500000, sample_mode=AcquisitionType.FINITE, samps_per_chan=10,
                                                active_edge=Edge.RISING)
                p = task.read(number_of_samples_per_channel=10)
            return sum(p) / len(p)
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
            self.tasks["clock"].co_channels.add_co_pulse_chan_freq(self.counter_channel, units=FrequencyUnits.HZ,
                                                                   idle_state=Level.LOW, initial_delay=0.0,
                                                                   freq=self.sample_rate, duty_cycle=self.duty_cycle)
            self.tasks["clock"].co_pulse_freq_timebase_src = '20MHzTimebase'
            self.tasks["clock"].co_pulse_freq_timebase_rate = self.clock_rate
            self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            self.tasks["clock"].co_pulse_term = self.clock[0]
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
            indices = [0, 1, 2, 3, 4, 5, 6]
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
            self.tasks["digital"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock[0],
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
            indices = [0, 1]
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
            self.tasks["piezo"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock[1],
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

    def write_galvo_sequences(self, galvo_sequences, indices=None):
        if indices is None:
            indices = [0, 1]
        if galvo_sequences.ndim > 1:
            n_channels, n_samples = galvo_sequences.shape
        else:
            n_channels = 1
            n_samples = galvo_sequences.shape[0]
        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping galvo sequences update.")
            return
        try:
            self.tasks["galvo"] = nidaqmx.Task("galvo")
            for ind in indices:
                self.tasks["galvo"].ao_channels.add_ao_voltage_chan(self.galvo_channels[ind], min_val=-10., max_val=10.)
            self.tasks["galvo"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock[0],
                                                           active_edge=Edge.RISING, sample_mode=self.mode,
                                                           samps_per_chan=n_samples)
            self.tasks["galvo"].write(galvo_sequences, auto_start=False)
            self._active["galvo"] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self, piezo_sequences=None, piezo_channels=None, galvo_sequences=None, galvo_channels=None,
                       digital_sequences=None, digital_channels=None, finite=True):
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
            if galvo_sequences is not None:
                self.write_galvo_sequences(galvo_sequences, indices=galvo_channels)
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

    def measure_ao(self, output_channels, input_channels, data, clk="/Dev1/ao/SampleClock"):
        if data.ndim > 1:
            _, num_samples = data.shape
        else:
            num_samples = data.shape[0]
        acquired_data = np.zeros(data.shape)
        with nidaqmx.Task() as output_task:
            output_task.ao_channels.add_ao_voltage_chan(output_channels, min_val=-10., max_val=10.)
            output_task.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                   sample_mode=AcquisitionType.FINITE,
                                                   samps_per_chan=num_samples)
            with nidaqmx.Task() as input_task:
                input_task.ai_channels.add_ai_voltage_chan(input_channels, min_val=-10., max_val=10.)
                input_task.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                      sample_mode=AcquisitionType.FINITE,
                                                      samps_per_chan=num_samples,
                                                      source=clk)
                writer = AnalogSingleChannelWriter(output_task.out_stream)
                reader = AnalogSingleChannelReader(input_task.in_stream)
                writer.write_many_sample(data)
                input_task.start()
                output_task.start()
                output_task.wait_until_done()
                input_task.wait_until_done()
                reader.read_many_sample(data=acquired_data, number_of_samples_per_channel=num_samples)
        return acquired_data

    def measure_do(self, output_channel, input_channel, data):
        num_samples = data.shape[0]
        with nidaqmx.Task() as task_do, nidaqmx.Task() as task_ai, nidaqmx.Task() as task_clock:
            task_clock.co_channels.add_co_pulse_chan_freq(self.counter_channel, units=FrequencyUnits.HZ,
                                                          idle_state=Level.LOW, initial_delay=0.0,
                                                          freq=self.sample_rate, duty_cycle=self.duty_cycle)
            task_clock.co_pulse_freq_timebase_src = '20MHzTimebase'
            task_clock.co_pulse_freq_timebase_rate = self.clock_rate
            task_clock.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            # Configure DO as before
            task_do.do_channels.add_do_chan(output_channel, line_grouping=LineGrouping.CHAN_PER_LINE)
            task_do.timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock,
                                               active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=num_samples)
            task_do.write(data == 1, auto_start=False)
            task_ai.ai_channels.add_ai_voltage_chan(input_channel)
            task_ai.timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock,
                                               active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=num_samples)
            # Start AI first but it waits for the trigger
            task_ai.start()
            # Trigger by writing to DO
            task_do.start()
            task_clock.start()
            task_do.wait_until_done()
            # Read the analog input response
            acquired_data = task_ai.read(number_of_samples_per_channel=num_samples, timeout=10)
            # Stop AI task
            task_clock.stop()
            task_do.stop()
            task_ai.stop()
        return acquired_data

    def check_task_status(self, task):
        try:
            if task.is_task_done():
                return True
            else:
                return False
        except nidaqmx.DaqError as e:
            self.logg.error(f"Error checking task status: {e}")
            return True
