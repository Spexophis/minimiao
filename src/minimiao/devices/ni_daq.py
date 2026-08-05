# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import warnings

import nidaqmx
import numpy as np
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, WAIT_INFINITELY, TaskMode, RegenerationMode
from nidaqmx.constants import PowerUpStates
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogMultiChannelWriter
from nidaqmx.system import System
from nidaqmx.types import DOPowerUpState

from minimiao import logger

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


def warm_up_native_library():
    with nidaqmx.Task():
        pass


class NIDAQ:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.devices = self._initialize()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()
        self.sample_rate = int(2.0e5)
        self.duty_cycle = float(0.5)
        self.analog_channels = ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2"]
        self.digital_channels = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line3", "Dev1/port0/line4",
                                 "Dev1/port0/line5", "Dev1/port0/line6", "Dev1/port0/line7"]
        self.led_channels = ["Dev2/port0/line0", "Dev2/port0/line1"]
        self.task_led = None
        self.clock_external_start_terminal = "/Dev1/PFI1"
        self.clock_counter_channels = ["/Dev1/ctr0", "/Dev2/ctr0"]
        self.clock_counter_terminals = ["/Dev1/PFI12", "/Dev2/PFI12"]
        self.run_mode = None
        self.retriggered = False
        self.sequence_samples = None

    def __del__(self):
        pass

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
                task.ai_channels.add_ai_voltage_chan("Dev1/ai0:2", min_val=-10.0, max_val=10.0)
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

    def write_dpc_sequences(self, led_sequences, indices=None, finite=True):

        if indices is None:
            indices = [0, 1]

        if finite:
            md = AcquisitionType.FINITE
        else:
            md = AcquisitionType.CONTINUOUS

        led_sequences = np.asarray(led_sequences)

        if led_sequences.ndim > 1:
            n_channels, n_samples = led_sequences.shape
            if n_channels == 1:
                led_sequences = led_sequences[0]
        else:
            n_channels = 1
            n_samples = led_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, skipping led sequences update.")
            return

        try:
            self.task_led = nidaqmx.Task()

            for ind in indices:
                self.task_led.do_channels.add_do_chan(self.led_channels[ind], line_grouping=LineGrouping.CHAN_PER_LINE)

            self.task_led.timing.cfg_samp_clk_timing(rate=8e6, active_edge=Edge.RISING,
                                                     sample_mode=md, samps_per_chan=n_samples)

            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.task_led.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            self.task_led.write(led_sequences == 1.0, auto_start=False)

            self.task_led.control(TaskMode.TASK_COMMIT)

            self.logg.info(f"LED sequence configured: "
                           f"{n_channels} channel(s), {n_samples} samples, "
                           f"clocked from {self.clock_counter_terminals[1]}.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

        except Exception as e:
            self.logg.error(f"Error configuring digit sequences: {e}")

    def run_dpc_sequences(self):
        self.task_led.start()

    def stop_dpc_sequences(self, cls=True):
        self.task_led.stop()
        if cls:
            self.task_led.close()

    def write_clock_channel(self, samples_per_trigger, trigger=False):
        """
        Configure the counter clock.

        If trigger=False:
            - samples_per_trigger > 0: generate one finite pulse train when started
            - samples_per_trigger <= 0: generate a continuous pulse train

        If trigger=True:
            - generate a finite pulse train once per external trigger
            - the task remains armed and retriggerable until stopped
        """
        try:
            # Close any previous clock task before recreating it
            if self.tasks.get("clock") is not None:
                try:
                    self.tasks["clock"].close()
                except Exception:
                    pass
                self.tasks["clock"] = None

            # Retriggerable pulse generation must be finite
            if trigger and samples_per_trigger <= 0:
                self.logg.error("Retriggerable clock generation requires samples_per_trigger > 0.")
                return

            self.tasks["clock"] = nidaqmx.Task("clock")

            # Counter pulse train
            co_channel = self.tasks["clock"].co_channels.add_co_pulse_chan_freq(counter=self.clock_counter_channels[0],
                                                                                freq=self.sample_rate,
                                                                                duty_cycle=self.duty_cycle)

            # Counter timing base
            co_channel.co_ctr_timebase_src = "20MHzTimebase"

            # Physical output terminal for the pulse train
            co_channel.co_pulse_term = self.clock_counter_terminals[0]

            # Timing mode
            if samples_per_trigger > 0:
                # Finite burst: exactly N pulses
                self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE,
                                                               samps_per_chan=samples_per_trigger)
            else:
                # Continuous pulse train
                self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)

            # External start trigger, optionally retriggerable
            if trigger:
                self.tasks["clock"].triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source=self.clock_external_start_terminal,
                    trigger_edge=Edge.RISING)

                self.tasks["clock"].triggers.start_trigger.retriggerable = True

            # Reserve resources and apply all routing/timing settings
            self.tasks["clock"].control(TaskMode.TASK_COMMIT)

            self._active["clock"] = True
            self._running["clock"] = False

            if trigger:
                self.logg.info(f"Retriggerable clock configured: "
                               f"{samples_per_trigger} pulses per external trigger.")
            elif samples_per_trigger > 0:
                self.logg.info(f"Finite clock configured: {samples_per_trigger} pulses.")
            else:
                self.logg.info("Continuous clock configured.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

        except Exception as e:
            self.logg.error(f"Error configuring clock channel: {e}")

    def write_digital_sequences(self, digital_sequences, indices=None):
        """
        Configure hardware-timed digital output.

        In retriggered sequence mode:
            - The digital task runs continuously with regeneration enabled.
            - It uses the retriggerable counter pulse train on PFI12 as its sample clock.
            - Each external trigger produces exactly one sequence because the counter
              generates exactly n_samples clock pulses per trigger.
        """
        if indices is None:
            indices = [2, 3, 4, 5, 6]

        digital_sequences = np.asarray(digital_sequences)

        if digital_sequences.ndim > 1:
            n_channels, n_samples = digital_sequences.shape
            if n_channels == 1:
                digital_sequences = digital_sequences[0]
        else:
            n_channels = 1
            n_samples = digital_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, "
                            "skipping digital sequences update.")
            return

        try:
            self.tasks["digital"] = nidaqmx.Task("digital")

            # Add digital output lines
            for ind in indices:
                self.tasks["digital"].do_channels.add_do_chan(self.digital_channels[ind],
                                                              line_grouping=LineGrouping.CHAN_PER_LINE)

            # Use retriggerable counter output as sample clock
            self.tasks["digital"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                             source=self.clock_counter_terminals[0],  # "/Dev1/PFI12"
                                                             active_edge=Edge.RISING,
                                                             sample_mode=self.run_mode,
                                                             samps_per_chan=n_samples)

            # For repeated trigger mode:
            # digital task is continuous, but advances only when PFI12 clocks arrive
            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.tasks["digital"].out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            # Write the digital waveform into the buffer, but do not start yet
            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)

            # Commit now so routing/timing errors appear at configuration time
            self.tasks["digital"].control(TaskMode.TASK_COMMIT)

            self._active["digital"] = True
            self._running["digital"] = False

            self.logg.info(f"Digital sequence configured: "
                           f"{n_channels} channel(s), {n_samples} samples, "
                           f"clocked from {self.clock_counter_terminals[0]}.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

        except Exception as e:
            self.logg.error(f"Error configuring digital sequences: {e}")

    def write_analog_sequences(self, analog_sequences, indices=None):
        """
        Configure hardware-timed analog output.

        In repeated-trigger mode:.
            - AO runs as a CONTINUOUS task with regeneration enabled.
            - AO uses /Dev2/PFI0 as its external sample clock.
            - /Dev2/PFI0 should receive the counter pulse train generated
              on /Dev1/PFI12.
            - Each external trigger causes the counter to emit exactly
              n_samples pulses, so exactly one AO sequence is output.
        """
        if indices is None:
            indices = [0, 1, 2]

        analog_sequences = np.asarray(analog_sequences)

        if analog_sequences.ndim > 1:
            n_channels, n_samples = analog_sequences.shape
        else:
            n_channels = 1
            n_samples = analog_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error("WARNING: Length of n_channels and indices differ, "
                            "skipping piezo sequences update.")
            return

        try:
            self.tasks["analog"] = nidaqmx.Task("analog")

            # Add AO channels
            for ind in indices:
                self.tasks["analog"].ao_channels.add_ao_voltage_chan(self.analog_channels[ind],
                                                                     min_val=0.0,
                                                                     max_val=10.0)

            # Use the retriggerable counter pulse train as AO sample clock.
            # Clock wiring: Dev1/PFI12 --> Dev2/PFI0
            self.tasks["analog"].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                            source=self.clock_counter_terminals[0],
                                                            active_edge=Edge.RISING,
                                                            sample_mode=self.run_mode,
                                                            samps_per_chan=n_samples)

            # In repeated-trigger mode, self.run_mode is CONTINUOUS.
            # Regeneration allows the same waveform buffer to be reused
            # every time a new burst of sample clocks arrives.
            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.tasks["analog"].out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            # Write waveform into the AO buffer, but do not start yet
            self.tasks["analog"].write(analog_sequences, auto_start=False)

            # Commit now so timing/routing errors appear during setup
            self.tasks["analog"].control(TaskMode.TASK_COMMIT)

            self._active["analog"] = True
            self._running["analog"] = False

            self.logg.info(f"Analog sequence configured: "
                           f"{n_channels} channel(s), {n_samples} samples, "
                           f"clocked from {self.clock_counter_terminals[0]}.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

        except Exception as e:
            self.logg.error(f"Error configuring analog sequences: {e}")

    def write_triggers(self,
                       digital_sequences=None, digital_channels=None,
                       analog_sequences=None, analog_channels=None,
                       finite=True, trg=False):
        """
        Configure synchronized counter-clocked AO/DO generation.

        Modes:
        ------------------------------------------------------------------
        finite=True, trg=False
            One finite AO/DO sequence starts when run_triggers() is called.

        finite=False, trg=False
            Continuous regenerated AO/DO generation with a continuous clock.

        finite=False, trg=True
            Repeated-trigger mode:
            one full AO/DO sequence is generated once per external trigger.
            This is implemented by:
                - retriggerable finite counter bursts
                - continuous regenerated AO/DO tasks
        ------------------------------------------------------------------
        """

        sequence_samples = None

        # ================================================================
        # Validate analog sequence
        # ================================================================
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
                self.logg.error("WARNING: Length of n_analog_channel and analog_channels differ, "
                                "skipping analog sequences.")
                return

            sequence_samples = analog_samples

        # ================================================================
        # Validate digital sequence
        # ================================================================
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
                self.logg.error("WARNING: Length of n_digital_channel and digital_channels differ, "
                                "skipping digital sequences.")
                return

            if sequence_samples is None:
                sequence_samples = digital_samples
            elif sequence_samples != digital_samples:
                self.logg.error("WARNING: Length of digital sequences and analog sequences differ.")
                return

        # ================================================================
        # Make sure something was provided
        # ================================================================
        if sequence_samples is None:
            self.logg.error("No analog or digital sequence was provided.")
            return

        self.sequence_samples = sequence_samples

        # ================================================================
        # Choose timing architecture
        # ================================================================
        if finite:
            # One finite sequence: Counter emits exactly sequence_samples pulses.
            self.run_mode = AcquisitionType.FINITE
            clock_samples = sequence_samples

            if trg:
                self.logg.warning("finite=True and trg=True uses your current triggered counter setup. "
                                  "For repeated one-sequence-per-trigger operation, use "
                                  "finite=False, trg=True.")

        else:
            # Continuous AO/DO tasks
            self.run_mode = AcquisitionType.CONTINUOUS

            if trg:
                # Repeated-trigger mode: Counter produces one finite pulse burst per external trigger.
                clock_samples = sequence_samples
            else:
                # Free-running continuous mode: Counter produces a continuous sample clock.
                clock_samples = 0

        # ================================================================
        # Configure counter sample clock first
        # ================================================================
        self.write_clock_channel(samples_per_trigger=clock_samples, trigger=trg)

        # ================================================================
        # Configure AO and DO tasks that consume that sample clock
        # ================================================================
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

            if self.run_mode == AcquisitionType.FINITE and not self.retriggered:
                for key, _task in self.tasks.items():
                    if key == "clock":
                        continue

                    if _task is None:
                        continue

                    if self._active.get(key, False):
                        _task.wait_until_done(WAIT_INFINITELY)
                        _task.stop()
                        self._running[key] = False

                if self.tasks.get("clock") is not None and self._running.get("clock", False):
                    self.tasks["clock"].stop()
                    self._running["clock"] = False

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
            co_channel = clk_task.co_channels.add_co_pulse_chan_freq(counter=self.clock_counter_channels[0],
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


def configure_do_power_up_high(device: str = "Dev2", high_lines: tuple[int, ...] = (5, 6, 7)) -> None:
    """
    Configure selected Port 0 lines to power up HIGH while preserving
    the existing power-up states of the other Port 0 lines.
    """
    system = System.local()

    existing_states = system.get_digital_power_up_states(device)

    existing_by_channel = {
        state.physical_channel: state.power_up_state
        for state in existing_states
    }

    port0_states = []

    for line_number in range(32):
        channel = f"{device}/port0/line{line_number}"

        if line_number in high_lines:
            power_up_state = PowerUpStates.HIGH
        else:
            # Preserve the current configuration of every other line.
            power_up_state = existing_by_channel.get(
                channel,
                PowerUpStates.TRISTATE,
            )

        port0_states.append(
            DOPowerUpState(
                physical_channel=channel,
                power_up_state=power_up_state,
            )
        )

    system.set_digital_power_up_states(
        device,
        port0_states,
    )
