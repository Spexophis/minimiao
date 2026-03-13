# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Cambridge Technology 6215H waveform generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from minimiao import logger


@dataclass(frozen=True)
class MotorSpecs:
    """
    Raw mechanical & electrical specifications of the 6215H galvanometer.
    Source: Cambridge Technology 62xxH Series datasheet DS00003 Rev A.
    """
    # --- Mechanical ---
    rotor_inertia_gcm2: float = 0.028  # gm·cm²  (±10%)
    rotor_inertia_kgm2: float = 2.80e-9  # kg·m²
    torque_constant_dyne_cm_per_A: float = 3.78e4  # dyne·cm/A  (±10%)
    torque_constant_Nm_per_A: float = 3.78e-3  # N·m/A  (SI)
    max_scan_angle_mech_deg: float = 20.0  # ±20° mechanical
    max_scan_angle_optical_deg: float = 40.0  # ±40° optical
    recommended_aperture_mm: tuple = (3.0, 7.0)

    # --- Electrical ---
    coil_resistance_ohm: float = 2.5  # Ω  (±10%)
    coil_inductance_uH: float = 94.0  # μH  (±10%)
    back_emf_uV_per_deg_per_s: float = 66.0  # μV/°/s  (±10%)
    back_emf_Vs_per_rad: float = 3.78e-3  # V·s/rad  (= Kt, as expected)

    # --- Current ratings ---
    rms_current_max_A: float = 4.1  # A  at Tcase = 50°C
    peak_current_max_A: float = 20.0  # A  absolute maximum

    # --- Thermal ---
    thermal_resistance_rotor_to_case_C_per_W: float = 1.0  # °C/W  (max)
    max_rotor_temp_C: float = 110.0  # °C

    # --- Step response (matched mirror, 0.1° step settled to 99%) ---
    step_response_3mm_mirror_us: float = 130.0
    step_response_5mm_mirror_us: float = 100.0

    # --- Position detector ---
    linearity_pct_over_20deg: float = 99.9  # %
    linearity_pct_over_40deg: float = 99.5  # %
    scale_drift_ppm_per_C: float = 50.0  # ppm/°C
    zero_drift_urad_per_C: float = 15.0  # μrad/°C
    repeatability_short_term_urad: float = 8.0  # μrad


@dataclass(frozen=True)
class DriverSpecs:
    """
    Raw specifications of the MicroMax 671 single-axis analog servo driver.
    Source: Cambridge Technology MicroMax 671XX datasheet & 670 manual.
    """
    supply_voltage_min_V: float = 15.0  # V  (per rail, e.g. ±15)
    supply_voltage_max_V: float = 28.0  # V  (per rail, e.g. ±28)
    peak_current_A: float = 10.0  # A
    rms_current_A: float = 5.0  # A  (supply & load dependent)

    # --- Analog I/O ---
    command_scale_V_per_mech_deg: float = 0.5  # V/°  (default)
    command_range_V: float = 10.0  # ±10 V
    position_output_scale_V_per_deg: float = 0.5  # V/° mechanical
    input_impedance_differential_kohm: float = 200.0
    input_impedance_single_ended_kohm: float = 100.0
    offset_range_V: float = 2.0  # ±2 V → ±4° offset

    # --- Stability ---
    temp_stability_ppm_per_C: float = 20.0  # ppm/°C
    operating_temp_range_C: tuple = (0.0, 50.0)


@dataclass(frozen=True)
class GalvoMirror:
    """
    Effective system parameters once a real mirror is mounted.
    Default values are for a ~3–5 mm matched mirror (published measurements).
    Reference: Schultz et al. (ResearchGate, 6215H + MicroMax 671 study)
               J_total ≈ 5.6×10⁻⁹ kg·m²
    """
    total_inertia_kgm2: float = 5.60e-9  # kg·m²  (rotor + mirror)
    mirror_size_mm: float = 3.0  # mm aperture (used for step response)


@dataclass
class GalvoSystem:
    """
    Complete specification and derived control limits for the
    Cambridge Technology 6215H galvanometer + MicroMax 671 driver system.
    """

    motor: MotorSpecs = field(default_factory=MotorSpecs)
    driver: DriverSpecs = field(default_factory=DriverSpecs)
    mirror: GalvoMirror = field(default_factory=GalvoMirror)

    @property
    def deg_per_volt_mechanical(self) -> float:
        """Mechanical degrees per volt of command input."""
        return 1.0 / self.driver.command_scale_V_per_mech_deg

    @property
    def deg_per_volt_optical(self) -> float:
        """Optical degrees per volt of command input."""
        return self.deg_per_volt_mechanical * 2.0

    @property
    def max_command_voltage_v(self) -> float:
        """Maximum safe command voltage (±)."""
        return self.driver.command_range_V

    @property
    def full_optical_fov_deg(self) -> float:
        """Total full-angle optical field of view."""
        return self.motor.max_scan_angle_optical_deg * 2.0

    @property
    def electrical_time_constant_s(self) -> float:
        """τ_e = L / R  — coil electrical time constant [s]."""
        return (self.motor.coil_inductance_uH * 1e-6) / self.motor.coil_resistance_ohm

    @property
    def electrical_bandwidth_hz(self) -> float:
        """Upper electrical bandwidth ceiling: f_e = 1 / (2π τ_e)  [Hz]."""
        return 1.0 / (2.0 * math.pi * self.electrical_time_constant_s)

    @property
    def peak_torque_Nm(self) -> float:
        """
        Peak torque limited by the *driver* peak current (10 A),
        which is lower than the galvo's own 20 A rating.
        T_peak = Kt × I_peak_driver
        """
        return self.motor.torque_constant_Nm_per_A * self.driver.peak_current_A

    @property
    def max_angular_accel_rad_s2(self) -> float:
        """α_max = T_peak / J_total  [rad/s²]."""
        return self.peak_torque_Nm / self.mirror.total_inertia_kgm2

    @property
    def max_angular_accel_deg_s2(self) -> float:
        """α_max in °/s²."""
        return math.degrees(self.max_angular_accel_rad_s2)

    def min_step_time_s(self, angle_mech_deg: float) -> float:
        """
        Theoretical minimum transit time for a mechanical angle move
        using constant-acceleration / constant-deceleration profile.

        t_min = 2 × √(θ / α_max)   [s]

        Note: actual settling adds ~2–3× on top for PID stabilisation.
        """
        theta_rad = math.radians(angle_mech_deg)
        return 2.0 * math.sqrt(theta_rad / self.max_angular_accel_rad_s2)

    def min_step_time_us(self, angle_mech_deg: float) -> float:
        """min_step_time_s in microseconds."""
        return self.min_step_time_s(angle_mech_deg) * 1e6

    @property
    def available_drive_voltage_v(self) -> float:
        """
        Voltage remaining for acceleration after resistive drop at RMS current.
        V_avail = V_supply_max − I_rms × R_coil
        """
        return (self.driver.supply_voltage_max_V
                - self.driver.rms_current_A * self.motor.coil_resistance_ohm)

    @property
    def max_angular_velocity_rad_s(self) -> float:
        """
        ω_max = V_available / K_BEMF   [rad/s]
        Back-EMF limited peak angular velocity.
        """
        return self.available_drive_voltage_v / self.motor.back_emf_Vs_per_rad

    @property
    def max_angular_velocity_deg_s(self) -> float:
        """ω_max in °/s."""
        return math.degrees(self.max_angular_velocity_rad_s)

    @property
    def max_command_velocity_V_per_us(self) -> float:
        """
        Maximum rate of change of the command voltage [V/μs].

        This is the back-EMF speed limit expressed in DAC output units —
        i.e. the steepest slope your command waveform may have.

        Derivation:
            ω_max  [°/s]  ×  scale [V/°]  ×  1e-6 [s/μs]  =  V/μs

        Any voltage ramp steeper than this value will cause the servo to
        saturate (back-EMF > available drive voltage), meaning the mirror
        will lag behind the command and the scan will be distorted.
        """
        return (self.max_angular_velocity_deg_s
                * self.driver.command_scale_V_per_mech_deg
                * 1e-6)

    @property
    def max_command_accel_V_per_us2(self) -> float:
        """
        Maximum rate of change of command voltage slope [V/μs²].

        This is the torque-limited angular acceleration expressed in DAC
        output units — i.e. the maximum curvature your command waveform
        may have.

        Derivation:
            α_max  [°/s²]  ×  scale [V/°]  ×  1e-12 [(s/μs)²]  =  V/μs²

        Exceeding this in a flyback or ramp segment means the requested
        torque is beyond what the driver peak current (10 A) can deliver,
        and the mirror will fall behind the trajectory.

        Practical use — minimum time for a linear voltage ramp of ΔV:
            t_min [μs] = 2 × √( ΔV / max_command_accel_V_per_us2 )
        """
        return (self.max_angular_accel_deg_s2
                * self.driver.command_scale_V_per_mech_deg
                * 1e-12)

    def min_ramp_time_us(self, delta_V: float) -> float:
        """
        Minimum time [μs] to slew the command voltage by delta_V [V]
        without exceeding the torque-limited acceleration.

        Equivalent to min_step_time_us() but expressed in voltage units:
            t_min = 2 × √( ΔV / max_command_accel_V_per_us2 )

        Parameters
        ----------
        delta_V : float
            Voltage swing to cover, e.g. full FOV = 2 × fov_v.
        """
        return 2.0 * math.sqrt(abs(delta_V) / self.max_command_accel_V_per_us2)

    def max_sine_frequency_Hz(self, half_amplitude_optical_deg: float) -> float:
        """
        Maximum sinusoidal scan frequency for a given ±optical amplitude [Hz].
        Derived from: ω_peak = 2π·f·θ_peak_rad ≤ ω_max_rad_s
        θ_peak_mech = half_amplitude_optical_deg / 2
        """
        theta_mech_rad = math.radians(half_amplitude_optical_deg / 2.0)
        return self.max_angular_velocity_rad_s / (2.0 * math.pi * theta_mech_rad)

    @property
    def rotor_power_dissipation_W(self) -> float:
        """
        Worst-case rotor Joule heating at maximum rated RMS current.
        P = I_rms² × R_coil
        """
        return (self.motor.rms_current_max_A ** 2) * self.motor.coil_resistance_ohm

    @property
    def max_case_temp_at_rated_rms_C(self) -> float:
        """
        Maximum allowable case temperature when running at full RMS current.
        T_case_max = T_rotor_max − P × R_thermal
        """
        delta_T = (self.rotor_power_dissipation_W
                   * self.motor.thermal_resistance_rotor_to_case_C_per_W)
        return self.motor.max_rotor_temp_C - delta_T

    def max_case_temp_at_current_C(self, i_rms_A: float) -> float:
        """Max allowable case temperature at an arbitrary RMS current [A]."""
        p = (i_rms_A ** 2) * self.motor.coil_resistance_ohm
        delta_T = p * self.motor.thermal_resistance_rotor_to_case_C_per_W
        return self.motor.max_rotor_temp_C - delta_T

    # Empirical closed-loop BW from published measurements on 6215H + MicroMax 671
    closed_loop_bandwidth_Hz: float = 1000.0  # Hz  (PID tuned, ~1 kHz)

    def phase_lag_deg(self, scan_freq_Hz: float) -> float:
        """
        First-order phase lag approximation at a given scan frequency.
        φ = arctan(f / BW_CL)   [degrees]
        """
        return math.degrees(math.atan(scan_freq_Hz / self.closed_loop_bandwidth_Hz))

    @property
    def min_dac_bits(self) -> int:
        """
        Minimum DAC bit depth so that one LSB ≤ short-term repeatability (8 μrad).
        Full range = ±20° mechanical = 698,132 μrad.
        """
        full_range_urad = math.radians(2.0 * self.motor.max_scan_angle_mech_deg) * 1e6
        steps = full_range_urad / self.motor.repeatability_short_term_urad
        return math.ceil(math.log2(steps))

    @property
    def lsb_voltage_mV(self) -> float:
        """
        Command voltage corresponding to the 8 μrad repeatability floor.
        V_lsb = (8μrad in °) × 0.5 V/° × 1000
        """
        angle_deg = math.degrees(self.motor.repeatability_short_term_urad * 1e-6)
        return angle_deg * self.driver.command_scale_V_per_mech_deg * 1000.0

    def zero_drift_at_dT_urad(self, delta_T_C: float) -> float:
        """Angular zero drift for a given temperature change [μrad]."""
        return self.motor.zero_drift_urad_per_C * delta_T_C

    def scale_drift_at_dT_pct(self, delta_T_C: float) -> float:
        """Scale factor drift [% of reading] for a given temperature change."""
        return (self.motor.scale_drift_ppm_per_C * delta_T_C) / 1e4

    @property
    def hard_limits(self) -> dict:
        """
        Dictionary of hard limits — values that must never be exceeded.
        Violating any of these risks servo shutdown or hardware damage.
        """
        return {
            "max_command_voltage_v": f"±{self.max_command_voltage_v:.1f} V",
            "max_scan_angle_mechanical_deg": f"±{self.motor.max_scan_angle_mech_deg:.1f}°",
            "max_scan_angle_optical_deg": f"±{self.motor.max_scan_angle_optical_deg:.1f}°",
            "max_rms_current_galvo_A": f"{self.motor.rms_current_max_A:.1f} A",
            "max_peak_current_driver_A": f"{self.driver.peak_current_A:.1f} A  (galvo rated: {self.motor.peak_current_max_A:.0f} A)",
            "max_supply_voltage_V": f"±{self.driver.supply_voltage_max_V:.0f} V",
            "max_case_temp_at_rated_rms_C": f"{self.max_case_temp_at_rated_rms_C:.1f} °C",
            "max_rotor_temp_C": f"{self.motor.max_rotor_temp_C:.0f} °C",
            "electrical_bw_ceiling_Hz": f"{self.electrical_bandwidth_hz:.0f} Hz",
            "practical_closed_loop_bw_Hz": f"~{self.closed_loop_bandwidth_Hz:.0f} Hz",
            "max_command_velocity_V_per_us": f"{self.max_command_velocity_V_per_us:.6f} V/μs",
            "max_command_accel_V_per_us2": f"{self.max_command_accel_V_per_us2:.4e} V/μs²",
        }


class GalvoWaveform:
    """
    Waveform generator for a servo galvo.
    """

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.gs: GalvoSystem = GalvoSystem()
        self.sample_rate_hz: float = 80e3
        self.n_pixels: int = 500
        self.n_lines: int = 500
        self.pixel_sx: float = -0.2
        self.pixel_sy: float = 0.2
        self.pixel_vx: float = 0.002
        self.pixel_vy: float = 0.00285
        self.fov_vx = self.pixel_vx * self.n_pixels
        self.fov_vy = self.pixel_vy * self.n_lines
        self.t_pixel_dwell_us = 1e6 / self.sample_rate_hz
        self.scan_velocity_v_us = self.pixel_vx / self.t_pixel_dwell_us
        self.t_ramp_us: float = 300.0
        self.ramp_acc_v_us2 = self.scan_velocity_v_us / self.t_ramp_us
        self.t_scan_us = self.t_pixel_dwell_us * self.n_pixels
        self.t_flyback_us: float = 510.0
        self.t_settle_us: float = 360.0
        self._validate()

    def set_params(self,
                   sample_rate_hz: float | None,
                   start_s: tuple | None, pixel_s: tuple | None, pixels_lines : tuple | None,
                   t_ramp_us : float | None, t_flyback_us : float | None, t_settle_us : float | None):
        """
        Set Scan Parameters
        """
        if sample_rate_hz is not None:
            self.sample_rate_hz = sample_rate_hz
            self.t_pixel_dwell_us = 1e6 / self.sample_rate_hz
        if start_s is not None:
            self.pixel_sx, self.pixel_sy = start_s
        if pixel_s is not None:
            self.pixel_vx, self.pixel_vy = pixel_s
            self.scan_velocity_v_us = self.pixel_vx / self.t_pixel_dwell_us
        if pixels_lines is not None:
            self.n_pixels, self.n_lines = pixels_lines
            self.t_scan_us = self.t_pixel_dwell_us * self.n_pixels
        self.fov_vx = self.pixel_vx * self.n_pixels
        self.fov_vy = self.pixel_vy * self.n_lines
        if t_ramp_us is not None:
            self.t_ramp_us  = t_ramp_us
            self.ramp_acc_v_us2 = self.scan_velocity_v_us / self.t_ramp_us
        if t_flyback_us is not None:
            self.t_flyback_us = t_flyback_us
        if t_settle_us is not None:
            self.t_settle_us = t_settle_us
        self._validate()

    @property
    def _fov_mech_deg(self) -> float:
        """Half-FOV in mechanical degrees (+peak to center)."""
        return self.fov_vx * self.gs.deg_per_volt_mechanical

    @property
    def _fov_optical_deg(self) -> float:
        """Half-FOV in optical degrees."""
        return self._fov_mech_deg * 2.0

    @property
    def _t_line_us(self) -> float:
        """
        Total period of one scan line:
        ramp → scan → flyback → settle
        """
        return self.t_ramp_us + self.t_scan_us + self.t_flyback_us + self.t_settle_us

    @property
    def _line_freq_hz(self) -> float:
        """Line repetition frequency [Hz]."""
        return 1e6 / self._t_line_us

    @property
    def _scan_velocity_deg_s(self) -> float:
        """
        Mean angular velocity during the active scan stroke [mech °/s].
        v = (2 × fov_mech_deg) / t_scan_s
        The factor 2 is because the mirror sweeps across the full FOV
        (−fov → +fov = 2 × fov_mech_deg total travel).
        """
        return (2.0 * self._fov_mech_deg) / (self.t_scan_us * 1e-6)

    @property
    def _scan_velocity_rad_s(self) -> float:
        return math.radians(self._scan_velocity_deg_s)

    @property
    def _flyback_velocity_deg_s(self) -> float:
        """
        Mean angular velocity during flyback [mech °/s].
        Flyback covers 2 × fov_mech_deg in t_flyback_us.
        """
        return (2.0 * self._fov_mech_deg) / (self.t_flyback_us * 1e-6)

    def _validate(self):
        """
        Validate all waveform parameters against the 6215H / MicroMax 671 limits.
        """
        m = self.gs
        errors = []
        warnings = []
        info = []

        # ── Derived quantities used throughout ─────────────────────────
        fov_mech = self._fov_mech_deg  # mechanical degrees (half-amplitude)
        fov_optical = self._fov_optical_deg  # optical degrees (half-amplitude)
        t_line_us = self._t_line_us
        line_freq = self._line_freq_hz
        scan_vel = self._scan_velocity_deg_s
        flyback_vel = self._flyback_velocity_deg_s

        for name, val in [
            ('sample_rate_hz', self.sample_rate_hz),
            ('fov_vx', self.fov_vx),
            ('fov_vx', self.fov_vy),
            ('n_pixels', self.n_pixels),
            ('n_lines', self.n_lines),
            ('t_pixel_dwell_us', self.t_pixel_dwell_us),
            ('t_scan_us', self.t_scan_us),
            ('t_flyback_us', self.t_flyback_us),
            ('t_ramp_us', self.t_ramp_us),
            ('t_settle_us', self.t_settle_us),
        ]:
            if val <= 0:
                errors.append(f"{name} must be > 0 (got {val}).")

        # t_pixel_dwell_us must match sample_rate_hz
        expected_dwell_us = 1e6 / self.sample_rate_hz
        if not math.isclose(self.t_pixel_dwell_us, expected_dwell_us, rel_tol=1e-4):
            errors.append(
                f"t_pixel_dwell_us ({self.t_pixel_dwell_us:.3f} μs) is inconsistent "
                f"with sample_rate_hz ({self.sample_rate_hz:.0f} Hz). "
                f"Expected {expected_dwell_us:.3f} μs."
            )

        # t_scan_us must equal n_pixels × t_pixel_dwell_us
        expected_scan_us = self.n_pixels * self.t_pixel_dwell_us
        if not math.isclose(self.t_scan_us, expected_scan_us, rel_tol=1e-4):
            errors.append(
                f"t_scan_us ({self.t_scan_us:.1f} μs) is inconsistent with "
                f"n_pixels × t_pixel_dwell_us = {expected_scan_us:.1f} μs."
            )

        # FOV voltage must not exceed driver command range
        if self.fov_vx + self.pixel_sx > m.max_command_voltage_v:
            errors.append(
                f"fov_vx ({self.fov_vx + self.pixel_sx:.2f} V) exceeds max driver command range "
                f"(±{m.max_command_voltage_v:.0f} V)."
            )

        if self.fov_vy + self.pixel_sy > m.max_command_voltage_v:
            errors.append(
                f"fov_vy ({self.fov_vy + self.pixel_sy:.2f} V) exceeds max driver command range "
                f"(±{m.max_command_voltage_v:.0f} V)."
            )

        # FOV mechanical angle must not exceed galvo max travel
        if fov_mech > m.motor.max_scan_angle_mech_deg:
            errors.append(
                f"fov_v maps to ±{fov_mech:.2f}° mechanical, which exceeds the "
                f"galvo max of ±{m.motor.max_scan_angle_mech_deg:.0f}°. "
                f"Reduce fov_v to ≤ {m.motor.max_scan_angle_mech_deg * m.driver.command_scale_V_per_mech_deg:.1f} V."
            )

        # Warn if FOV is > 90 % of max range (risk of over-position trip)
        fov_utilisation = fov_mech / m.motor.max_scan_angle_mech_deg
        if 0.9 < fov_utilisation <= 1.0:
            warnings.append(
                f"fov_v uses {fov_utilisation * 100:.1f}% of the galvo's angular range. "
                f"Leave at least 10% headroom to avoid over-position faults."
            )

        max_vel_deg_s = m.max_angular_velocity_deg_s

        if scan_vel > max_vel_deg_s:
            errors.append(
                f"Scan velocity ({scan_vel:.0f} °/s mech) exceeds back-EMF limited "
                f"max of {max_vel_deg_s:.0f} °/s. "
                f"Reduce fov_v or increase t_scan_us."
            )
        elif scan_vel > 0.8 * max_vel_deg_s:
            warnings.append(
                f"Scan velocity ({scan_vel:.0f} °/s) is {scan_vel / max_vel_deg_s * 100:.0f}% "
                f"of the back-EMF limit ({max_vel_deg_s:.0f} °/s). "
                f"Drive may saturate on acceleration at line boundaries."
            )

        if flyback_vel > max_vel_deg_s:
            errors.append(
                f"Flyback mean velocity ({flyback_vel:.0f} °/s mech) exceeds "
                f"back-EMF limited max of {max_vel_deg_s:.0f} °/s. "
                f"Increase t_flyback_us."
            )

        # Flyback must complete a full 2×fov_mech_deg swing.
        # Minimum transit time is the bang-bang limit; add 2× for PID settling.
        min_flyback_transit_us = m.min_step_time_us(2.0 * fov_mech)
        min_flyback_with_settle_us = min_flyback_transit_us * 2.5  # empirical safety factor

        if self.t_flyback_us < min_flyback_transit_us:
            errors.append(
                f"t_flyback_us ({self.t_flyback_us:.0f} μs) is shorter than the "
                f"theoretical minimum bang-bang transit time "
                f"({min_flyback_transit_us:.0f} μs) for a {2 * fov_mech:.2f}° swing. "
                f"The servo cannot physically complete this move."
            )
        elif self.t_flyback_us < min_flyback_with_settle_us:
            warnings.append(
                f"t_flyback_us ({self.t_flyback_us:.0f} μs) is below the recommended "
                f"minimum including settling ({min_flyback_with_settle_us:.0f} μs). "
                f"Position may not be settled at scan start, causing line shear."
            )

        # After flyback the mirror must ring down before pixel acquisition.
        # Minimum settle ≈ 2.5× the small-angle 99%-step-response time.
        step_resp_us = m.motor.step_response_3mm_mirror_us  # 130 μs (conservative)
        min_settle_us = step_resp_us * 2.5  # ≈ 325 μs

        if self.t_settle_us < step_resp_us:
            errors.append(
                f"t_settle_us ({self.t_settle_us:.0f} μs) is shorter than the "
                f"small-angle step response ({step_resp_us:.0f} μs). "
                f"Mirror oscillation will corrupt the first pixels of every line."
            )
        elif self.t_settle_us < min_settle_us:
            warnings.append(
                f"t_settle_us ({self.t_settle_us:.0f} μs) is below the recommended "
                f"2.5× step-response margin ({min_settle_us:.0f} μs). "
                f"First pixels may be slightly distorted."
            )

        # t_ramp must be long enough to slew from rest to the start-of-scan
        # position (full fov_mech travel from −fov to +fov edge).
        min_ramp_us = m.min_step_time_us(fov_mech) * 2.0

        if self.t_ramp_us < min_ramp_us:
            warnings.append(
                f"t_ramp_us ({self.t_ramp_us:.0f} μs) may be insufficient to slew to "
                f"the scan start position ({fov_mech:.2f}° mechanical). "
                f"Recommended minimum: {min_ramp_us:.0f} μs."
            )

        # Phase lag at the line scan frequency must be acceptable.
        # >45° means the servo is already at its −3 dB point.
        phase_lag = m.phase_lag_deg(line_freq)

        if line_freq > m.closed_loop_bandwidth_Hz:
            errors.append(
                f"Line frequency ({line_freq:.1f} Hz) exceeds the closed-loop "
                f"bandwidth ({m.closed_loop_bandwidth_Hz:.0f} Hz). "
                f"The servo cannot track the command waveform. "
                f"Reduce scan speed or increase t_line."
            )
        elif phase_lag > 25.0:
            warnings.append(
                f"Phase lag at line frequency ({line_freq:.1f} Hz) is {phase_lag:.1f}°. "
                f"Above 25° geometric distortion becomes significant. "
                f"Consider phase pre-compensation or reducing line frequency."
            )

        # Pixel dwell below ~1 μs means DAC update period is < servo response.
        if self.t_pixel_dwell_us < 1.0:
            warnings.append(
                f"t_pixel_dwell_us ({self.t_pixel_dwell_us:.2f} μs) is very short. "
                f"The galvo position may shift significantly within a single pixel."
            )

        # # Maximum practical sample rate: respect electrical BW ceiling
        # if self.sample_rate_hz > m.electrical_bandwidth_hz:
        #     warnings.append(
        #         f"sample_rate_hz ({self.sample_rate_hz / 1e3:.1f} kHz) exceeds the "
        #         f"coil electrical bandwidth ceiling ({m.electrical_bandwidth_hz:.0f} Hz). "
        #         f"Current commands above this frequency are attenuated by the coil L/R."
        #     )

        t_frame_ms = (t_line_us * self.n_lines) / 1e3
        frame_rate = 1e3 / t_frame_ms if t_frame_ms > 0 else 0.0

        info.append(f"FOV                  : ±{fov_mech:.2f}° mech / ±{fov_optical:.2f}° optical")
        info.append(f"Pixel dwell          : {self.t_pixel_dwell_us:.2f} μs")
        info.append(f"Scan time per line   : {self.t_scan_us:.1f} μs")
        info.append(f"Full line period     : {t_line_us:.1f} μs  (ramp+scan+flyback+settle)")
        info.append(f"Line frequency       : {line_freq:.2f} Hz")
        info.append(f"Phase lag @ f_line   : {phase_lag:.1f}°")
        info.append(f"Scan velocity        : {scan_vel:.0f} °/s mech  ({scan_vel / max_vel_deg_s * 100:.1f}% of limit)")
        info.append(f"Flyback velocity     : {flyback_vel:.0f} °/s mech")
        info.append(f"Total frame time     : {t_frame_ms:.2f} ms  →  {frame_rate:.2f} fps")
        info.append(f"Min flyback transit  : {min_flyback_transit_us:.0f} μs  (bang-bang)")
        info.append(f"Min flyback+settle   : {min_flyback_with_settle_us:.0f} μs  (recommended)")


        for msg in info:
            self.logg.info(f"[galvo validate] {msg}")
        for msg in warnings:
            self.logg.warning(f"[galvo validate] WARNING: {msg}")
        for msg in errors:
            self.logg.error(f"[galvo validate] ERROR: {msg}")

        if warnings:
            print(
                f"GalvoWaveform has {len(warnings)} warnings(s):\n"
                + "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(warnings))
            )

        if errors:
            raise ValueError(
                f"GalvoWaveform has {len(errors)} hard error(s):\n"
                + "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(errors))
            )

    def build_scan(self):
        sr = self.sample_rate_hz
        dt = 1.0 / sr
        dt_us = dt * 1e6

        ramp_samps = int(np.ceil(self.t_ramp_us * 1e-6 * sr))
        scan_samps = int(np.ceil(self.t_scan_us * 1e-6 * sr))
        back_samps = int(np.ceil(self.t_flyback_us * 1e-6 * sr))

        ramp_t = np.arange(ramp_samps) * dt_us
        ramp_curve = 0.5 * self.ramp_acc_v_us2 * ramp_t ** 2
        offset = self.pixel_sx - ramp_curve[-1]
        ramp_curve += offset
        scan_t = np.arange(scan_samps) * dt_us
        scan_curve = self.scan_velocity_v_us * scan_t + self.pixel_sx
        back_t = np.arange(back_samps) * dt_us
        back_curve, vel, acc = ramp_curve[0] + self.get_trajectory(back_t, p0=scan_curve[-1] - ramp_curve[0], v0=self.scan_velocity_v_us)
        one_period_x = np.hstack((ramp_curve, scan_curve, back_curve))
        fast_axis = np.tile(one_period_x, self.n_lines)

        step_curve, vel, acc = self.get_trajectory(back_t, p0=self.pixel_vy, v0=0)
        one_period_y = np.hstack((np.zeros(ramp_samps + scan_samps), step_curve[::-1]))
        slow_axis = (one_period_y + self.pixel_vy * np.arange(self.n_lines)[:, None]).ravel()
        back_curve, vel, acc = self.get_trajectory(back_t, p0=slow_axis[-back_samps], v0=0)
        slow_axis[-back_samps:] = back_curve
        slow_axis += self.pixel_sy

        gate = np.hstack((np.zeros(ramp_samps), np.ones(scan_samps), np.zeros(back_samps)))
        gate_ttl = np.tile(gate, self.n_lines)

        return fast_axis, slow_axis, gate_ttl

    def get_trajectory(self,
                       t: np.ndarray,
                       p0: float = 0.0, v0: float = 0.0,
                       p1: float = 0.0, v1: float = 0.0, a1: float = 0.0):
        """
        Compute a minimum-jerk (5th-order polynomial) trajectory from
        (p0, v0, a=0) to (p1, v1, a1) over the time array t.

        Parameters
        ----------
        t     : 1-D ndarray, monotonically increasing.
        p0    : initial position.
        v0    : initial velocity  [position_unit / time_unit].
        p1    : end position      (default 0).
        v1    : end velocity      (default 0).
        a1    : end acceleration  (default 0).
        """
        a_max = self.gs.max_command_accel_V_per_us2  # maximum allowed acceleration
        v_max = self.gs.max_command_velocity_V_per_us  # maximum allowed velocity

        T = t[-1] - t[0]
        tau = (t - t[0]) / T

        # Boundary values in normalised (tau) units
        V0 = v0 * T
        V1 = v1 * T
        A1 = a1 * T ** 2
        # a0 = 0 always → c2 = 0

        # Auxiliary quantities
        P = p1 - p0 - V0  # position residual after linear term
        V = V1 - V0  # velocity residual
        A = A1  # acceleration residual (A0=0)

        # Closed-form coefficients (unique solution to the 6 BCs)
        c0 = p0
        c1 = V0
        c2 = 0.0
        c3 = 10 * P - 4 * V + 0.5 * A
        c4 = -15 * P + 7 * V - A
        c5 = 6 * P - 3 * V + 0.5 * A

        pos = c0 + c1 * tau + c2 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
        vel = (c1 + 2 * c2 * tau + 3 * c3 * tau ** 2 + 4 * c4 * tau ** 3 + 5 * c5 * tau ** 4) / T
        acc = (2 * c2 + 6 * c3 * tau + 12 * c4 * tau ** 2 + 20 * c5 * tau ** 3) / T ** 2

        peak_v = float(np.max(np.abs(vel)))
        peak_a = float(np.max(np.abs(acc)))

        violations = []
        if peak_v > v_max:
            T_min_v = T * (peak_v / v_max)
            violations.append(f"velocity {peak_v:.4g} > v_max {v_max:.4g}  (need T >= {T_min_v:.4g})")
        if peak_a > a_max:
            T_min_a = T * np.sqrt(peak_a / a_max)
            violations.append(f"acceleration {peak_a:.4g} > a_max {a_max:.4g}  (need T >= {T_min_a:.4g})")
        if violations:
            raise ValueError("Trajectory constraint violated: " + "; ".join(violations))

        return pos, vel, acc
