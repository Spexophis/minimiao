# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
PI P-737.1SL PIFOC specimen Z positioner waveform generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from minimiao import logger


@dataclass(frozen=True)
class PiezoActuatorSpecs:
    """
    Raw specifications of the PI P-737.1SL PIFOC specimen Z positioner.
    Source: Physik Instrumente P-737 series datasheet.
    """
    # --- Travel & resolution ---
    travel_range_um: float = 100.0          # μm  (open-loop)
    resolution_nm: float = 0.4              # nm  (closed-loop, minimum incremental motion)
    repeatability_nm: float = 2.0           # nm  (closed-loop, bidirectional)
    linearity_pct: float = 0.03             # %   (closed-loop)

    # --- Dynamics ---
    resonant_frequency_unloaded_hz: float = 310.0   # Hz  (no specimen)
    resonant_frequency_loaded_hz: float = 160.0     # Hz  (50 g specimen)
    settling_time_ms: float = 10.0                  # ms  (closed-loop, to 1% error)

    # --- Load & mechanics ---
    max_load_g: float = 50.0        # g
    push_pull_force_n: float = 20.0 # N  maximum applicable force
    tilting_urad_per_um: float = 2.0 # μrad/μm  (pitch/yaw, open-loop)

    # --- Sensor ---
    sensor_type: str = "Capacitive"  # integrated capacitive sensor (closed-loop)

    # --- Thermal ---
    operating_temp_range_c: tuple = (10.0, 50.0)


@dataclass(frozen=True)
class PiezoControllerSpecs:
    """
    Analog control specifications compatible with PI E-709 / E-516 controllers
    used with the P-737.1SL.
    Source: PI E-709 datasheet.
    """
    # --- Voltage I/O ---
    input_voltage_min_v: float = 0.0    # V  (maps to 0 μm)
    input_voltage_max_v: float = 10.0   # V  (maps to 100 μm)
    output_voltage_min_v: float = -2.0  # V  (sensor monitor output minimum)
    output_voltage_max_v: float = 12.0  # V  (sensor monitor output maximum)

    # --- Scale ---
    um_per_volt: float = 10.0           # μm/V  (100 μm / 10 V)
    volt_per_um: float = 0.1            # V/μm

    # --- Controller bandwidth ---
    closed_loop_bandwidth_hz: float = 100.0  # Hz  (typical, PID-tuned)
    notch_filter_freq_hz: float = 310.0      # Hz  (matched to unloaded resonance)

    # --- Noise ---
    output_noise_rms_mv: float = 0.3    # mV RMS  position noise voltage


@dataclass
class PiezoSystem:
    """
    Complete specification and derived limits for the
    PI P-737.1SL positioner + analog controller system.
    """

    actuator: PiezoActuatorSpecs = field(default_factory=PiezoActuatorSpecs)
    controller: PiezoControllerSpecs = field(default_factory=PiezoControllerSpecs)

    @property
    def travel_range_v(self) -> float:
        """Full travel range expressed in command voltage [V]."""
        return (self.actuator.travel_range_um
                * self.controller.volt_per_um)

    @property
    def resolution_v(self) -> float:
        """Minimum incremental motion expressed in command voltage [V]."""
        return (self.actuator.resolution_nm * 1e-3
                * self.controller.volt_per_um)

    @property
    def settling_time_s(self) -> float:
        """Closed-loop settling time [s]."""
        return self.actuator.settling_time_ms * 1e-3

    @property
    def min_step_time_s(self) -> float:
        """
        Minimum safe dwell time per Z plane to allow settling [s].
        Based on 3× the settling time for conservative margin.
        """
        return self.actuator.settling_time_ms * 3e-3

    @property
    def max_slew_rate_v_per_s(self) -> float:
        """
        Maximum command slew rate [V/s] limited by closed-loop bandwidth.
        Approximated as: 2π × BW × travel_range_v / 2
        (peak velocity of a sinusoid at closed-loop BW covering full range)
        """
        return (2.0 * math.pi
                * self.controller.closed_loop_bandwidth_hz
                * self.travel_range_v / 2.0)

    def min_return_time_s(self, delta_um: float) -> float:
        """
        Minimum return/slew time [s] to move delta_um without exceeding
        the closed-loop bandwidth-limited slew rate.
        """
        delta_v = abs(delta_um) * self.controller.volt_per_um
        slew = self.max_slew_rate_v_per_s
        if slew == 0:
            return 0.0
        return delta_v / slew

    def position_to_voltage(self, position_um: float) -> float:
        """Convert absolute position [μm] to command voltage [V]."""
        return position_um * self.controller.volt_per_um

    def voltage_to_position(self, voltage_v: float) -> float:
        """Convert command voltage [V] to absolute position [μm]."""
        return voltage_v * self.controller.um_per_volt

    @property
    def hard_limits(self) -> dict:
        """
        Dictionary of hard limits — values that must never be exceeded.
        """
        return {
            "min_command_voltage_v": f"{self.controller.input_voltage_min_v:.1f} V",
            "max_command_voltage_v": f"{self.controller.input_voltage_max_v:.1f} V",
            "travel_range_um": f"{self.actuator.travel_range_um:.0f} μm",
            "max_load_g": f"{self.actuator.max_load_g:.0f} g",
            "settling_time_ms": f"{self.actuator.settling_time_ms:.0f} ms",
            "closed_loop_bandwidth_hz": f"{self.controller.closed_loop_bandwidth_hz:.0f} Hz",
            "resonant_frequency_loaded_hz": f"{self.actuator.resonant_frequency_loaded_hz:.0f} Hz",
        }


class PiezoWaveform:
    """
    Waveform generator for the PI P-737.1SL PIFOC specimen Z positioner.

    Generates the analog voltage sequence to step through Z planes during
    a 3-D scan. For each plane the piezo holds a constant voltage while
    the galvo acquires the XY frame, then ramps smoothly to the next
    position during the inter-plane return interval.
    """

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.ps: PiezoSystem = PiezoSystem()
        self.sample_rate_hz: float = 80e3

        # Z scan parameters (in physical units; converted to voltage internally)
        self.z_center_um: float = 50.0          # μm  center of Z stack
        self.z_range_um: float = 3.2            # μm  total Z travel
        self.z_step_um: float = 0.16            # μm  step between planes
        self.z_return_time_s: float = 0.04      # s   inter-plane move time

        self._update_derived()

    def _update_derived(self):
        self.z_start_um = self.z_center_um - self.z_range_um / 2.0
        self.n_planes = max(1, int(np.ceil(self.z_range_um / self.z_step_um)))
        self.z_positions_um = (self.z_start_um
                               + self.z_step_um * np.arange(self.n_planes))
        self.z_positions_v = (self.z_positions_um
                              * self.ps.controller.volt_per_um)
        self.return_samples = int(np.ceil(self.z_return_time_s
                                          * self.sample_rate_hz))

    def set_params(self,
                   sample_rate_hz: float | None = None,
                   z_center_um: float | None = None,
                   z_range_um: float | None = None,
                   z_step_um: float | None = None,
                   z_return_time_s: float | None = None):
        """Update scan parameters and re-validate."""
        if sample_rate_hz is not None:
            self.sample_rate_hz = sample_rate_hz
        if z_center_um is not None:
            self.z_center_um = z_center_um
        if z_range_um is not None:
            self.z_range_um = z_range_um
        if z_step_um is not None:
            self.z_step_um = z_step_um
        if z_return_time_s is not None:
            self.z_return_time_s = z_return_time_s
        self._update_derived()
        self._validate()

    def _validate(self):
        """Validate Z scan parameters against P-737.1SL limits."""
        ps = self.ps
        errors = []
        warnings = []
        info = []

        for name, val in [
            ('sample_rate_hz', self.sample_rate_hz),
            ('z_range_um', self.z_range_um),
            ('z_step_um', self.z_step_um),
            ('z_return_time_s', self.z_return_time_s),
        ]:
            if val <= 0:
                errors.append(f"{name} must be > 0 (got {val}).")

        # Z start and end must be within travel range
        z_end_um = self.z_start_um + self.z_range_um
        if self.z_start_um < 0.0:
            errors.append(
                f"Z start ({self.z_start_um:.2f} μm) is below the travel range minimum (0 μm)."
            )
        if z_end_um > ps.actuator.travel_range_um:
            errors.append(
                f"Z end ({z_end_um:.2f} μm) exceeds the travel range "
                f"({ps.actuator.travel_range_um:.0f} μm)."
            )

        # Return time must be sufficient for settling
        min_return = ps.settling_time_s
        if self.z_return_time_s < min_return:
            errors.append(
                f"z_return_time_s ({self.z_return_time_s * 1e3:.1f} ms) is shorter than "
                f"the minimum settling time ({min_return * 1e3:.0f} ms)."
            )
        elif self.z_return_time_s < min_return * 2.0:
            warnings.append(
                f"z_return_time_s ({self.z_return_time_s * 1e3:.1f} ms) is less than "
                f"2× the settling time. Consider increasing for reliable settling."
            )

        # Step should be larger than resolution
        if self.z_step_um < ps.actuator.resolution_nm * 1e-3:
            errors.append(
                f"z_step_um ({self.z_step_um * 1e3:.1f} nm) is below the sensor "
                f"resolution ({ps.actuator.resolution_nm:.1f} nm)."
            )

        # Slew rate check for return move
        min_slew_return = ps.min_return_time_s(self.z_step_um)
        if self.z_return_time_s < min_slew_return:
            warnings.append(
                f"z_return_time_s ({self.z_return_time_s * 1e3:.1f} ms) may be too short "
                f"to slew one step ({self.z_step_um:.3f} μm) within closed-loop BW "
                f"(recommended ≥ {min_slew_return * 1e3:.2f} ms)."
            )

        frame_rate = 1.0 / (self.z_return_time_s * self.n_planes) if self.n_planes > 0 else 0.0

        info.append(f"Z center             : {self.z_center_um:.2f} μm")
        info.append(f"Z range              : {self.z_range_um:.3f} μm "
                    f"({self.z_start_um:.3f} → {self.z_start_um + self.z_range_um:.3f} μm)")
        info.append(f"Z step               : {self.z_step_um:.4f} μm  "
                    f"({self.z_step_um * 1e3:.2f} nm)")
        info.append(f"Z planes             : {self.n_planes}")
        info.append(f"Return time          : {self.z_return_time_s * 1e3:.1f} ms")
        info.append(f"Voltage range        : "
                    f"{self.z_positions_v[0]:.4f} – {self.z_positions_v[-1]:.4f} V")
        info.append(f"Est. Z-stack rate    : {frame_rate:.3f} Hz  (return-only, no galvo dwell)")

        for msg in info:
            self.logg.info(f"[piezo validate] {msg}")
        for msg in warnings:
            self.logg.warning(f"[piezo validate] WARNING: {msg}")
        for msg in errors:
            self.logg.error(f"[piezo validate] ERROR: {msg}")

        if warnings:
            print(
                f"PiezoWaveform has {len(warnings)} warning(s):\n"
                + "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(warnings))
            )

        if errors:
            raise ValueError(
                f"PiezoWaveform has {len(errors)} hard error(s):\n"
                + "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(errors))
            )

    def build_z_steps(self,
                      samples_per_plane: int,
                      z_positions_v: np.ndarray | None = None,
                      return_samples: int | None = None) -> np.ndarray:
        """
        Build the piezo voltage sequence for a Z-stack acquisition.

        For each plane the piezo holds at the target voltage for
        `samples_per_plane` samples, then ramps smoothly (linear interpolation)
        to the next position in `return_samples` samples. After the last plane
        the piezo returns to the first position.

        Parameters
        ----------
        samples_per_plane : int
            Number of DAQ samples during which the piezo is held constant
            (equals the galvo frame length, excluding the return interval).
        z_positions_v : ndarray, optional
            Voltage set-points for each Z plane. Defaults to
            ``self.z_positions_v``.
        return_samples : int, optional
            Samples for the inter-plane ramp. Defaults to
            ``self.return_samples``.

        Returns
        -------
        ndarray, shape (n_planes × (samples_per_plane + return_samples),)
            Analog voltage sequence for the piezo DAQ channel.
        """
        if z_positions_v is None:
            z_positions_v = self.z_positions_v
        if return_samples is None:
            return_samples = self.return_samples

        n_planes = len(z_positions_v)
        total = n_planes * (samples_per_plane + return_samples)
        seq = np.empty(total)

        for iz, v in enumerate(z_positions_v):
            base = iz * (samples_per_plane + return_samples)
            seq[base: base + samples_per_plane] = v

            next_v = z_positions_v[(iz + 1) % n_planes]
            seq[base + samples_per_plane: base + samples_per_plane + return_samples] = (
                np.linspace(v, next_v, return_samples, endpoint=False)
            )

        return seq

    def build_z_ramp(self,
                     z_start_v: float,
                     z_end_v: float,
                     total_samples: int) -> np.ndarray:
        """
        Generate a smooth sinusoidal ramp from ``z_start_v`` to ``z_end_v``
        over ``total_samples`` samples.

        The ramp uses a half-sine acceleration profile so that velocity is
        zero at both endpoints, reducing mechanical shock on the piezo and
        specimen.
        """
        tau = np.linspace(0.0, np.pi, total_samples)
        return z_start_v + (z_end_v - z_start_v) * (1.0 - np.cos(tau)) / 2.0
