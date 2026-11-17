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
        self.tasks = {"trigger": None, "clock": None,
                      "digital_dev1": None, "digital_dev2": None,
                      "analog_dev1": None, "analog_dev2": None}
        self._active = {key: False for key in self.tasks.keys()}
        self._running = {key: False for key in self.tasks.keys()}
        self.sample_rate = int(2.0e5)
        self.digital_dev2_sample_rate = 8e6  # DPC LEDs: Dev2 DO runs on its own onboard sample clock
        self.duty_cycle = float(0.5)
        self.analog_channels_dev1 = ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2", "Dev1/ao3"]
        self.analog_channels_dev2 = ["Dev2/ao0", "Dev2/ao1", "Dev2/ao2"]
        self.digital_channels_dev1 = ["Dev1/port0/line0", "Dev1/port0/line1", "Dev1/port0/line3", "Dev1/port0/line4",
                                      "Dev1/port0/line5", "Dev1/port0/line6", "Dev1/port0/line7"]
        self.digital_channels_dev2 = ["Dev2/port0/line0", "Dev2/port0/line1"]
        self.trigger_counter_channels = ["/Dev1/ctr1", "/Dev2/ctr1"]
        self.trigger_counter_terminals = ["/Dev1/PFI2", "/Dev2/PFI2"]
        self.trigger_delay = float(1.0)
        self.trigger_duty_cycle = float(0.01)
        self.clock_external_start_terminals = ["/Dev1/PFI1", "/Dev2/PFI1"]
        self.clock_counter_channels = ["/Dev1/ctr0", "/Dev2/ctr0"]
        self.clock_counter_terminals = ["/Dev1/PFI12", "/Dev2/PFI12"]
        self.run_mode = None
        self.retriggered = False
        self.sequence_samples = None

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

    def _close_task(self, key):
        """Stop and close the task stored under key (if any) and release its resources."""
        task = self.tasks.get(key)
        if task is not None:
            try:
                task.close()
            except Exception as e:
                self.logg.warning(f"Closing task {key}: {e}")
        self.tasks[key] = None
        self._active[key] = False
        self._running[key] = False

    def set_piezo_position(self, pos, indices=None):
        if indices is None:
            indices = [0, 1, 2]
        if len(pos) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping piezo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.analog_channels_dev2[ind], min_val=0., max_val=10.)
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

    def set_galvo_position(self, pos, indices=None):
        if indices is None:
            indices = [0, 1]
        if len(pos) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping galvo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.analog_channels_dev1[ind], min_val=-10., max_val=10.)
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
                task.ai_channels.add_ai_voltage_chan("Dev2/ai0:1", min_val=-10.0, max_val=10.0)  # galvo x, y
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

    def set_retarder_phase(self, phs, indices=None):
        if indices is None:
            indices = [2, 3]
        if len(phs) != len(indices):
            self.logg.error("WARNING: Length of pos and indices differ, skipping piezo position update.")
            return
        try:
            with nidaqmx.Task() as task:
                for ind in indices:
                    task.ao_channels.add_ao_voltage_chan(self.analog_channels_dev1[ind], min_val=0., max_val=5.)
                task.write(phs)
                task.wait_until_done(WAIT_INFINITELY)
                task.stop()
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_trigger_counter(self, delay):
        """
        Set the trigger counter frequency (Hz) and duty cycle, independent of the sample clock.
        Takes effect on the next write_triggers(..., trg=True).
        """
        if delay <= 0:
            self.logg.error("Trigger delay must be > 0")
            return
        self.trigger_delay = float(delay)

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

            # Close any previous trigger task before recreating it
            if self.tasks.get("trigger") is not None:
                try:
                    self.tasks["trigger"].close()
                except Exception:
                    pass
                self.tasks["trigger"] = None

            # Retriggerable pulse generation must be finite
            if trigger and samples_per_trigger <= 0:
                self.logg.error("Retriggerable clock generation requires samples_per_trigger > 0.")
                return

            if trigger:
                # The retriggerable clock ignores triggers during a burst,
                # so the trigger period must be longer than one sequence.
                trigger_period = self.trigger_delay + samples_per_trigger / self.sample_rate
                trigger_rate = 1 / trigger_period

                self.tasks["trigger"] = nidaqmx.Task("trigger")

                trg_channel = self.tasks["trigger"].co_channels.add_co_pulse_chan_freq(
                    counter=self.trigger_counter_channels[0],
                    freq=trigger_rate,
                    duty_cycle=self.trigger_duty_cycle)

                trg_channel.co_ctr_timebase_src = "20MHzTimebase"
                trg_channel.co_pulse_term = self.trigger_counter_terminals[0]

                self.tasks["trigger"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)

                self.tasks["trigger"].control(TaskMode.TASK_COMMIT)

                self._active["trigger"] = True
                self._running["trigger"] = False

                self.logg.info(
                    f"Trigger counter configured: {trigger_rate} Hz, output on {self.trigger_counter_terminals[0]}.")

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
                    trigger_source=self.clock_external_start_terminals[0],
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

    def write_digital_sequences_dev1(self, digital_sequences, indices=None):
        """
        Hardware-timed digital output on Dev1 (lasers, camera, SLM triggers).
        indices select entries of self.digital_channels_dev1; clocked from /Dev1/PFI12.
        """
        if indices is None:
            indices = [2, 3, 4, 5, 6]
        self._write_digital_task("digital_dev1", self.digital_channels_dev1, indices, digital_sequences,
                                 clock_source=self.clock_counter_terminals[0])

    def write_digital_sequences_dev2(self, digital_sequences, indices=None):
        """
        Hardware-timed digital output on Dev2 (DPC LEDs on port0/line0, line1).
        indices select entries of self.digital_channels_dev2. This task runs on Dev2's onboard sample clock at
        self.digital_dev2_sample_rate (8 MHz), not on the shared counter clock, so it is not sample-synchronous
        with the other tasks and its sequence length is independent of theirs.
        """
        if indices is None:
            indices = [0, 1]
        self._write_digital_task("digital_dev2", self.digital_channels_dev2, indices, digital_sequences,
                                 clock_source=None, rate=self.digital_dev2_sample_rate)

    def _write_digital_task(self, key, channels, indices, digital_sequences, clock_source, rate=None):
        """clock_source None: the device's onboard sample clock at rate (default self.sample_rate)."""
        """
        In retriggered sequence mode:
            - The digital task runs continuously with regeneration enabled.
            - It uses the retriggerable counter pulse train as its sample clock.
            - Each external trigger produces exactly one sequence because the counter
              generates exactly n_samples clock pulses per trigger.
        """
        digital_sequences = np.asarray(digital_sequences)

        if digital_sequences.ndim > 1:
            n_channels, n_samples = digital_sequences.shape
            if n_channels == 1:
                digital_sequences = digital_sequences[0]
        else:
            n_channels = 1
            n_samples = digital_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error(f"WARNING: Length of n_channels and indices differ, skipping {key} sequences update.")
            return

        try:
            self._close_task(key)
            self.tasks[key] = nidaqmx.Task(key)

            # Add digital output lines
            for ind in indices:
                self.tasks[key].do_channels.add_do_chan(channels[ind], line_grouping=LineGrouping.CHAN_PER_LINE)

            # Sample clock: the counter pulse train, or the onboard clock when clock_source is None
            rate = self.sample_rate if rate is None else rate
            timing = dict(rate=rate, active_edge=Edge.RISING, sample_mode=self.run_mode, samps_per_chan=n_samples)
            if clock_source is not None:
                timing["source"] = clock_source
            self.tasks[key].timing.cfg_samp_clk_timing(**timing)

            # For repeated trigger mode:
            # digital task is continuous, but advances only when clock pulses arrive
            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.tasks[key].out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            # Write the digital waveform into the buffer, but do not start yet
            self.tasks[key].write(digital_sequences == 1.0, auto_start=False)

            # Commit now so routing/timing errors appear at configuration time
            self.tasks[key].control(TaskMode.TASK_COMMIT)

            self._active[key] = True
            self._running[key] = False

            self.logg.info(f"Digital sequence configured: {[channels[i] for i in indices]}, {n_samples} samples, "
                           f"clocked from {clock_source or f'the onboard clock at {rate:g} Hz'}.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

        except Exception as e:
            self._close_task(key)  # do not leave a half-configured task holding the lines
            self.logg.error(f"Error configuring {key} sequences: {e}")

    def write_analog_sequences_dev1(self, analog_sequences, indices=None):
        """
        Hardware-timed analog output on Dev1 (galvo x/y on ao0/ao1, phase retarder on ao2/ao3).
        indices select entries of self.analog_channels_dev1; clocked from /Dev1/PFI12.
        """
        if indices is None:
            indices = [0, 1]
        self._write_analog_task("analog_dev1", self.analog_channels_dev1, indices, analog_sequences,
                                clock_source=self.clock_counter_terminals[0], v_range=(-10.0, 10.0))

    def write_analog_sequences_dev2(self, analog_sequences, indices=None):
        """
        Hardware-timed analog output on Dev2 (piezo x/y/z on ao0/ao1/ao2, 0-10 V).
        indices select entries of self.analog_channels_dev2; clocked from /Dev2/PFI12, which receives the same
        counter pulse train as /Dev1/PFI12, so this task stays sample-synchronous with the Dev1 tasks.
        """
        if indices is None:
            indices = [2]
        self._write_analog_task("analog_dev2", self.analog_channels_dev2, indices, analog_sequences,
                                clock_source=self.clock_counter_terminals[1], v_range=(0.0, 10.0))

    def _write_analog_task(self, key, channels, indices, analog_sequences, clock_source, v_range):
        """
        In repeated-trigger mode:
            - AO runs as a CONTINUOUS task with regeneration enabled.
            - Each external trigger causes the counter to emit exactly n_samples pulses,
              so exactly one AO sequence is output.
        """
        analog_sequences = np.asarray(analog_sequences)

        if analog_sequences.ndim > 1:
            n_channels, n_samples = analog_sequences.shape
            if n_channels == 1:
                analog_sequences = analog_sequences[0]
        else:
            n_channels = 1
            n_samples = analog_sequences.shape[0]

        if n_channels != len(indices):
            self.logg.error(f"WARNING: Length of n_channels and indices differ, skipping {key} sequences update.")
            return

        try:
            self._close_task(key)
            self.tasks[key] = nidaqmx.Task(key)

            # Add AO channels
            for ind in indices:
                self.tasks[key].ao_channels.add_ao_voltage_chan(channels[ind], min_val=v_range[0], max_val=v_range[1])

            # Use the counter pulse train as AO sample clock
            self.tasks[key].timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                       source=clock_source,
                                                       active_edge=Edge.RISING,
                                                       sample_mode=self.run_mode,
                                                       samps_per_chan=n_samples)

            # In repeated-trigger mode, self.run_mode is CONTINUOUS.
            # Regeneration allows the same waveform buffer to be reused
            # every time a new burst of sample clocks arrives.
            if self.run_mode == AcquisitionType.CONTINUOUS:
                self.tasks[key].out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            # Write waveform into the AO buffer, but do not start yet
            self.tasks[key].write(analog_sequences, auto_start=False)

            # Commit now so timing/routing errors appear during setup
            self.tasks[key].control(TaskMode.TASK_COMMIT)

            self._active[key] = True
            self._running[key] = False

            self.logg.info(f"Analog sequence configured: {[channels[i] for i in indices]}, "
                           f"{n_samples} samples, clocked from {clock_source}.")

        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

        except Exception as e:
            self._close_task(key)  # do not leave a half-configured task reserving the AO channels
            self.logg.error(f"Error configuring {key} sequences: {e}")

    def write_triggers(self,
                       digital_sequences_dev1=None, digital_channels_dev1=None,
                       digital_sequences_dev2=None, digital_channels_dev2=None,
                       analog_sequences_dev1=None, analog_channels_dev1=None,
                       analog_sequences_dev2=None, analog_channels_dev2=None,
                       finite=True, trg=False):
        """
        Configure synchronized counter-clocked AO/DO generation.

        digital_sequences_dev1 / digital_channels_dev1: Dev1 DO, indices into digital_channels_dev1.
        digital_sequences_dev2 / digital_channels_dev2: Dev2 DO (DPC LEDs), indices into digital_channels_dev2;
            runs on Dev2's onboard clock at digital_dev2_sample_rate, so its length is independent.
        analog_sequences_dev1 / analog_channels_dev1: Dev1 AO (galvo, retarder), indices into analog_channels_dev1.
        analog_sequences_dev2 / analog_channels_dev2: Dev2 AO (piezo), indices into analog_channels_dev2.
        The Dev1 digital and both analog sequences share the counter clock and must have the same number of samples.
        With only digital_sequences_dev2 given, no counter clock is configured.

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
        # Validate sequences: channel counts, and one length for the counter-clocked ones
        # ================================================================
        sequences = (("Dev1 digital", digital_sequences_dev1, digital_channels_dev1),
                     ("Dev2 digital", digital_sequences_dev2, digital_channels_dev2),
                     ("Dev1 analog", analog_sequences_dev1, analog_channels_dev1),
                     ("Dev2 analog", analog_sequences_dev2, analog_channels_dev2))
        for name, seq, chs in sequences:
            if seq is None:
                continue
            seq = np.asarray(seq)
            n_channel, n_samples = seq.shape if seq.ndim > 1 else (1, seq.shape[0])

            if chs is not None and n_channel != len(chs):
                self.logg.error(f"WARNING: Length of {name} sequences and channels differ, skipping sequences.")
                return

            if name == "Dev2 digital":
                continue  # own onboard clock, length independent of the others

            if sequence_samples is None:
                sequence_samples = n_samples
            elif sequence_samples != n_samples:
                self.logg.error(f"WARNING: Length of {name} sequences ({n_samples}) differs from the other "
                                f"sequences ({sequence_samples}).")
                return

        def _n_channels(seq):
            return np.asarray(seq).shape[0] if np.ndim(seq) > 1 else 1

        if digital_sequences_dev1 is not None and digital_channels_dev1 is None:
            digital_channels_dev1 = list(range(_n_channels(digital_sequences_dev1)))
        if digital_sequences_dev2 is not None and digital_channels_dev2 is None:
            digital_channels_dev2 = list(range(_n_channels(digital_sequences_dev2)))
        if analog_sequences_dev1 is not None and analog_channels_dev1 is None:
            analog_channels_dev1 = list(range(_n_channels(analog_sequences_dev1)))
        if analog_sequences_dev2 is not None and analog_channels_dev2 is None:
            analog_channels_dev2 = [2]

        # ================================================================
        # Make sure something was provided
        # ================================================================
        if sequence_samples is None and digital_sequences_dev2 is None:
            self.logg.error("No analog or digital sequence was provided.")
            return

        self.sequence_samples = sequence_samples

        if trg and digital_sequences_dev2 is not None:
            self.logg.warning("Dev2 digital (DPC) runs on its own onboard clock and is not retriggered.")

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
        # Configure counter sample clock first (not needed when only Dev2 digital is used)
        # ================================================================
        if sequence_samples is not None:
            self.write_clock_channel(samples_per_trigger=clock_samples, trigger=trg)

        # ================================================================
        # Configure AO and DO tasks that consume that sample clock
        # ================================================================
        try:
            if digital_sequences_dev1 is not None:
                self.write_digital_sequences_dev1(digital_sequences_dev1, indices=digital_channels_dev1)

            if digital_sequences_dev2 is not None:
                self.write_digital_sequences_dev2(digital_sequences_dev2, indices=digital_channels_dev2)

            if analog_sequences_dev1 is not None:
                self.write_analog_sequences_dev1(analog_sequences_dev1, indices=analog_channels_dev1)

            if analog_sequences_dev2 is not None:
                self.write_analog_sequences_dev2(analog_sequences_dev2, indices=analog_channels_dev2)

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
                if key in ("clock", "trigger"):
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
            if self.tasks["clock"] is None and not any(self._active.get(k, False) for k in self.tasks):
                self.logg.error("No trigger task has been configured.")
                return

            self.start_triggers()

            # the counter clock is absent when only Dev2 digital (onboard clock) is used
            if self.tasks["clock"] is not None and not self._running.get("clock", False):
                self.tasks["clock"].start()
                self._running["clock"] = True

            if self.tasks.get("trigger") is not None and not self._running.get("trigger", False):
                self.tasks["trigger"].start()
                self._running["trigger"] = True

            self.logg.info("Trigger is running")

            if self.run_mode == AcquisitionType.FINITE and not self.retriggered:
                for key, _task in self.tasks.items():
                    if key in ("clock", "trigger"):
                        continue

                    if _task is None:
                        continue

                    if self._active.get(key, False):
                        _task.wait_until_done(WAIT_INFINITELY)
                        _task.stop()
                        self._running[key] = False

                if self.tasks.get("trigger") is not None and self._running.get("trigger", False):
                    self.tasks["trigger"].stop()
                    self._running["trigger"] = False

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
        if self.tasks.get("trigger") is not None and self._running.get("trigger", False):
            try:
                self.tasks["trigger"].stop()
            except nidaqmx.DaqWarning as e:
                self.logg.warning("DaqWarning caught as exception: %s", e)
            self._running["trigger"] = False

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

            try:
                _task.close()
            except nidaqmx.DaqWarning as e:
                self.logg.warning("DaqWarning caught as exception: %s", e)
            except Exception as e:
                self.logg.warning(f"Closing task {key}: {e}")

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


def configure_do_power_up_high(device: str = "Dev2",
                               low_lines: tuple[int, ...] = (5, 6, 7),
                               high_lines: tuple[int, ...] = (5, 6, 7)) -> None:
    """
    Configure selected Port 0 lines to power up HIGH while preserving
    the existing power-up states of the other Port 0 lines.
    """
    system = System.local()

    existing_states = system.get_digital_power_up_states(device)

    existing_by_channel = {state.physical_channel: state.power_up_state for state in existing_states}

    port0_states = []

    for line_number in range(32):
        channel = f"{device}/port0/line{line_number}"

        if line_number in high_lines:
            power_up_state = PowerUpStates.HIGH
        elif line_number in low_lines:
            power_up_state = PowerUpStates.LOW
        else:
            power_up_state = existing_by_channel.get(channel, PowerUpStates.TRISTATE)

        port0_states.append(DOPowerUpState(physical_channel=channel, power_up_state=power_up_state))

    system.set_digital_power_up_states(device, port0_states)
