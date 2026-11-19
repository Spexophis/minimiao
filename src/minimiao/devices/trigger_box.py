# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Python control for the Triggerbox (SAMD21G18A, firmware v2.0).

Protocol (spec v2.0, sec. 7): ASCII over USB-UART, 115200 8N1, CR-terminated.
    COMMAND                      ARM, RUN, STOP, RESET, STATUS, DIGSTAT3, ...
    COMMAND:type:v1;v2;...       DIG1:16:300;1300;200;100;1
Replies are "OK: ..." or "ERR N: ...".
"""

import math
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import serial
from serial.tools import list_ports

from minimiao import logger

# Hardware limits (spec v2.0)
F_CPU_HZ = 48_000_000
MAX_SYSCLK_HZ = 30_000  # MAX_SYSTEM_CLOCK_HZ
SAFE_SYSCLK_HZ = 20_000  # SAFE_SYSTEM_CLOCK_HZ (firmware warns above)
MAX_PULSE_FREQ_HZ = 20_000  # ERR 1 range
N_DIGITAL = 8
N_ANALOG = 4
N_OUTPUTS = 8
PWM_CHANNELS = (1, 2, 3, 4, 5)  # TCC1, TCC2, TC3, TC4, TC5; DIG6-8 are trigger-only per spec
MAX_MV = 5000
MAX_COM2_MSGS = 4
MAX_COM2_LEN = 10
UINT16_MAX = 65535

ERROR_CODES = {
    1: "ERR_INVALID_FREQ",
    2: "ERR_INVALID_TIME",
    3: "ERR_INVALID_PULSE",
    4: "ERR_INVALID_VOLTAGE",
    5: "ERR_INVALID_CHANNEL",
    6: "ERR_INVALID_OUTPUT",
    7: "ERR_STOP_BEFORE_START",
    8: "ERR_SYSTEM_RUNNING",
    9: "ERR_INVALID_PARAM",
    10: "ERR_INVALID_MODE",
    11: "ERR_PULSE_WIDTH_TOO_SHORT",
    12: "ERR_PULSE_WIDTH_TOO_LONG",
    13: "ERR_NOT_ARMED",
    14: "COM2TX_ERR_BUFFER_FULL",
    15: "ERR_CMD_UNKNOWN",
}


class TriggerBoxError(RuntimeError):
    def __init__(self, code: int, message: str, command: str):
        self.code = code
        self.command = command
        name = ERROR_CODES.get(code, "UNKNOWN")
        super().__init__(f"{command!r} -> ERR {code} ({name}): {message}")


# Channel definitions
@dataclass
class DigitalPulse:
    """PWM pulse train: DIGn:16:start;stop;freq;width;out"""
    channel: int
    start_ms: int
    stop_ms: int
    freq_hz: int
    width_us: int
    output: Optional[int] = None  # defaults to the channel number
    enabled: bool = True

    kind = "dig"

    @property
    def out(self) -> int:
        return self.channel if self.output is None else self.output

    def command(self) -> str:
        return (f"DIG{self.channel}:16:{self.start_ms};{self.stop_ms};"
                f"{self.freq_hz};{self.width_us};{self.out}")


@dataclass
class DigitalTrigger:
    """Level output, HIGH from start to stop: DIGTRIGn:16:start;stop;out"""
    channel: int
    start_ms: int
    stop_ms: int
    output: Optional[int] = None
    enabled: bool = True

    kind = "dig"

    @property
    def out(self) -> int:
        return self.channel if self.output is None else self.output

    def command(self) -> str:
        return f"DIGTRIG{self.channel}:16:{self.start_ms};{self.stop_ms};{self.out}"


@dataclass
class AnalogFixed:
    """Fixed voltage from start to stop, 0 V after: ANAn:16:start;stop;mV"""
    channel: int
    start_ms: int
    stop_ms: int
    mv: int
    enabled: bool = True

    kind = "ana"

    def command(self) -> str:
        return f"ANA{self.channel}:16:{self.start_ms};{self.stop_ms};{self.mv}"


@dataclass
class AnalogRamp:
    """Linear ramp: ANARAMPn:16:start;stop;start_mV;stop_mV"""
    channel: int
    start_ms: int
    stop_ms: int
    start_mv: int
    stop_mv: int
    enabled: bool = True

    kind = "ana"

    def command(self) -> str:
        return (f"ANARAMP{self.channel}:16:{self.start_ms};{self.stop_ms};"
                f"{self.start_mv};{self.stop_mv}")


@dataclass
class Com2Message:
    """Scheduled text on the secondary UART: COM2TX:s:<8-digit ms><text>"""
    time_ms: int
    text: str

    def command(self) -> str:
        return f"COM2TX:s:{self.time_ms:08d}{self.text}"


Channel = Union[DigitalPulse, DigitalTrigger, AnalogFixed, AnalogRamp]


# Sequence
@dataclass
class Sequence:
    sysclk_hz: int = 1000
    external_trigger: bool = False
    max_time_ms: Optional[int] = None
    channels: list = field(default_factory=list)
    com2: list = field(default_factory=list)

    # ---- building -------------------------------------------------------
    def add(self, *items: Union[Channel, Com2Message]) -> "Sequence":
        for it in items:
            (self.com2 if isinstance(it, Com2Message) else self.channels).append(it)
        return self

    @property
    def duration_ms(self) -> int:
        stops = [c.stop_ms for c in self.channels if c.enabled]
        return max(stops, default=0)

    @property
    def tick_us(self) -> float:
        return 1e6 / self.sysclk_hz

    def commands(self) -> list[str]:
        """Full command list, excluding RESET/ARM/RUN."""
        cmds = [f"SYSCLK:16:{self.sysclk_hz}",
                f"TRIG:8:{1 if self.external_trigger else 0}"]
        if self.max_time_ms is not None:
            cmds.append(f"MAXTIME:32:{self.max_time_ms}")
        for c in self.channels:
            cmds.append(c.command())
            prefix = "DIG" if c.kind == "dig" else "ANA"
            cmds.append(f"{prefix}{c.channel}EN:8:{int(c.enabled)}")
        cmds += [m.command() for m in self.com2]
        return cmds

    # ---- validation -----------------------------------------------------
    def validate(self, strict: bool = True) -> list[str]:
        """Check against spec limits. Raises ValueError on errors (if strict);
        returns and emits warnings for things that are allowed but risky."""
        err, warn = [], []
        f = self.sysclk_hz
        if not 1 <= f <= MAX_SYSCLK_HZ:
            err.append(f"SYSCLK {f} Hz outside 1-{MAX_SYSCLK_HZ} Hz")
        elif f > SAFE_SYSCLK_HZ:
            warn.append(f"SYSCLK {f} Hz > {SAFE_SYSCLK_HZ} Hz safe limit; "
                        "ticks may be missed with many channels active")

        seen = set()
        for c in self.channels:
            tag = f"{type(c).__name__}(ch{c.channel})"
            key = (c.kind, c.channel)
            if key in seen:
                err.append(f"{tag}: channel defined twice")
            seen.add(key)

            nmax = N_DIGITAL if c.kind == "dig" else N_ANALOG
            if not 1 <= c.channel <= nmax:
                err.append(f"{tag}: channel must be 1-{nmax}")
            if c.start_ms < 0 or c.stop_ms <= c.start_ms:
                err.append(f"{tag}: need 0 <= start < stop")
            if c.stop_ms > UINT16_MAX:
                warn.append(f"{tag}: times > {UINT16_MAX} ms sent with type code 16; "
                            "check that firmware parses them as uint32")
            if self.max_time_ms is not None and c.stop_ms > self.max_time_ms:
                err.append(f"{tag}: stop {c.stop_ms} ms > MAXTIME {self.max_time_ms} ms")
            if f < 1000 and ((c.start_ms * f) % 1000 or (c.stop_ms * f) % 1000):
                warn.append(f"{tag}: start/stop not on a tick boundary at {f} Hz")

            if c.kind == "dig" and not 1 <= c.out <= N_OUTPUTS:
                err.append(f"{tag}: output must be 1-{N_OUTPUTS}")

            if isinstance(c, DigitalPulse):
                if c.channel not in PWM_CHANNELS:
                    warn.append(f"{tag}: spec lists DIG6-8 as trigger-only (no PWM timer)")
                if not 1 <= c.freq_hz <= MAX_PULSE_FREQ_HZ:
                    err.append(f"{tag}: freq must be 1-{MAX_PULSE_FREQ_HZ} Hz")
                    continue
                period_us = 1e6 / c.freq_hz
                if not 0 < c.width_us <= UINT16_MAX:
                    err.append(f"{tag}: width must be 1-{UINT16_MAX} us")
                if c.width_us >= period_us:
                    err.append(f"{tag}: width {c.width_us} us >= period {period_us:.1f} us")
                if c.width_us < 2 * self.tick_us:
                    err.append(f"{tag}: width {c.width_us} us < 2 ticks "
                               f"({2 * self.tick_us:.1f} us at {f} Hz)")
                if f % c.freq_hz:
                    warn.append(f"{tag}: {c.freq_hz} Hz does not divide SYSCLK {f} Hz; "
                                "pulses not phase-locked to ticks")

            if isinstance(c, AnalogFixed):
                if not 0 <= c.mv <= MAX_MV:
                    err.append(f"{tag}: voltage must be 0-{MAX_MV} mV")
            if isinstance(c, AnalogRamp):
                for v in (c.start_mv, c.stop_mv):
                    if not 0 <= v <= MAX_MV:
                        err.append(f"{tag}: voltage must be 0-{MAX_MV} mV")

        # GCLK4 prescaler feasibility (spec 9.6): one integer divider 1..1024
        # must give 100..0xFFFFFF counts for SYSCLK and 100..65535 for every PWM.
        pwm = [c for c in self.channels if isinstance(c, DigitalPulse) and c.enabled
               and 1 <= c.freq_hz <= MAX_PULSE_FREQ_HZ]
        if 1 <= f <= MAX_SYSCLK_HZ and gclk4_prescaler(f, [c.freq_hz for c in pwm]) is None:
            warn.append("No common GCLK4 prescaler fits SYSCLK and all PWM frequencies; "
                        "ARM will likely return ERR 3. Spread of PWM frequencies too wide.")

        if len(self.com2) > MAX_COM2_MSGS:
            err.append(f"COM2TX: max {MAX_COM2_MSGS} messages")
        for m in self.com2:
            if not 1 <= len(m.text) <= MAX_COM2_LEN:
                err.append(f"COM2TX '{m.text}': length must be 1-{MAX_COM2_LEN}")
            if not 0 <= m.time_ms <= 99_999_999:
                err.append(f"COM2TX '{m.text}': time must fit 8 digits")
        if self.com2 and f > 1000:
            warn.append("COM2TX with other channels is only guaranteed at slow SYSCLK")

        for w in warn:
            warnings.warn(w, stacklevel=2)
        if err and strict:
            raise ValueError("Invalid sequence:\n  " + "\n  ".join(err))
        return err + warn

    # ---- preview --------------------------------------------------------
    def waveforms(self, max_samples: int = 200_000):
        """Expected outputs sampled on the tick grid (decimated if long).
        Returns t_ms, {output: bool array}, {analog_ch: mV array}."""
        n_ticks = int(math.ceil(self.duration_ms * self.sysclk_hz / 1000)) + 1
        step = max(1, n_ticks // max_samples)
        ticks = np.arange(0, n_ticks, step)
        t_ms = ticks * 1000.0 / self.sysclk_hz

        dig = {o: np.zeros(ticks.size, bool) for o in range(1, N_OUTPUTS + 1)}
        ana = {}
        for c in self.channels:
            if not c.enabled:
                continue
            active = (t_ms >= c.start_ms) & (t_ms < c.stop_ms)
            if isinstance(c, DigitalTrigger):
                dig[c.out] |= active
            elif isinstance(c, DigitalPulse):
                phase_us = ((t_ms - c.start_ms) * 1000.0) % (1e6 / c.freq_hz)
                dig[c.out] |= active & (phase_us < c.width_us)  # OR semantics (spec 5.1)
            elif isinstance(c, AnalogFixed):
                ana[c.channel] = np.where(active, c.mv, 0.0)
            elif isinstance(c, AnalogRamp):
                frac = np.clip((t_ms - c.start_ms) / (c.stop_ms - c.start_ms), 0, 1)
                ana[c.channel] = np.where(active, c.start_mv + frac * (c.stop_mv - c.start_mv), 0.0)
        return t_ms, dig, ana

    def plot(self, ax=None, show: bool = True):
        import matplotlib.pyplot as plt
        t, dig, ana = self.waveforms()
        used = [o for o, v in dig.items() if v.any()]
        rows = len(used) + (1 if ana else 0)
        fig, axes = plt.subplots(max(rows, 1), 1, sharex=True,
                                 figsize=(9, 1.0 + 0.8 * rows), squeeze=False)
        axes = axes[:, 0]
        for a, o in zip(axes, used):
            a.step(t, dig[o].astype(int), where="post", lw=0.8)
            a.set_ylabel(f"OUT{o}", rotation=0, ha="right", va="center")
            a.set_yticks([])
            a.set_ylim(-0.2, 1.2)
        if ana:
            a = axes[-1]
            for ch, v in ana.items():
                a.plot(t, v / 1000, label=f"ANA{ch}")
            a.set_ylabel("V")
            a.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (ms)")
        fig.suptitle(f"SYSCLK {self.sysclk_hz} Hz, tick {self.tick_us:.1f} us", fontsize=10)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


def gclk4_prescaler(sysclk_hz: float, pwm_freqs: list[float]) -> Optional[int]:
    """Lowest integer GCLK4 divider (1..1024) satisfying spec 9.6, or None."""
    lo = F_CPU_HZ / (0xFFFFFF * sysclk_hz)
    hi = F_CPU_HZ / (100 * sysclk_hz)
    for fp in pwm_freqs:
        lo = max(lo, F_CPU_HZ / (UINT16_MAX * fp))
        hi = min(hi, F_CPU_HZ / (100 * fp))
    p = max(1, math.ceil(lo))
    return p if p <= min(hi, 1024) else None


def choose_sysclk(pulse_freqs: list[int], min_width_us: float = 0.0,
                  max_hz: int = SAFE_SYSCLK_HZ, widths_us: Optional[list] = None) -> Optional[int]:
    """
    Highest SYSCLK (<= max_hz) that is a multiple of LCM(pulse_freqs) (pulses
    phase-locked to ticks), has a common GCLK4 prescaler with every PWM
    frequency (spec 9.6), and makes the shortest pulse >= 2 ticks.
    E.g. a 10 Hz train needs SYSCLK <= ~6.5 kHz because of the shared prescaler.
    """
    l = 1
    for fq in pulse_freqs:
        l = l * fq // math.gcd(l, fq)
    top = (max_hz // l) * l
    # first pass: every pulse width is a whole number of ticks; second pass: any feasible clock
    for exact in ((True, False) if widths_us else (False,)):
        f = top
        while f >= l and f >= 1:
            if (gclk4_prescaler(f, pulse_freqs) is not None
                    and (min_width_us <= 0 or min_width_us >= 2e6 / f)
                    and (not exact or all((w * f) % 1_000_000 == 0 for w in widths_us))):
                return f
            f -= l
    return None


@dataclass
class PulseTrain:
    """
    N identical TTL pulses on one TriggerBox channel.

    interval_ms is onset-to-onset (e.g. 100 ms -> 10 Hz). The TriggerBox
    generates it as one PWM window (DIGn); the window stops halfway into the
    gap after the last pulse, so the last pulse always completes and no
    extra pulse can start.
    """
    n_pulses: int = 5
    width_ms: float = 10.0
    interval_ms: float = 100.0
    start_ms: int = 0
    channel: int = 1
    output: Optional[int] = None  # defaults to the channel number

    def __post_init__(self):
        errs = []
        if self.n_pulses < 1:
            errs.append("n_pulses must be >= 1")
        if not 0 < self.width_ms < self.interval_ms:
            errs.append(f"need 0 < width ({self.width_ms} ms) < interval ({self.interval_ms} ms)")
        f = 1000.0 / self.interval_ms if self.interval_ms > 0 else 0
        if self.interval_ms > 0 and abs(f - round(f)) > 1e-9:
            errs.append(f"interval {self.interval_ms} ms = {f:.4g} Hz; the TriggerBox needs a whole-number "
                        f"frequency, i.e. interval = 1000/N ms (100, 50, 25, 20, ...)")
        if self.channel not in PWM_CHANNELS:
            errs.append(f"channel must be one of {PWM_CHANNELS} (PWM-capable)")
        if self.start_ms < 0 or self.start_ms != int(self.start_ms):
            errs.append("start_ms must be a whole number >= 0")
        if errs:
            raise ValueError("Invalid pulse train: " + "; ".join(errs))

    @property
    def freq_hz(self) -> int:
        return int(round(1000.0 / self.interval_ms))

    @property
    def width_us(self) -> int:
        return int(round(self.width_ms * 1000))

    @property
    def onsets_ms(self) -> list:
        return [self.start_ms + k * self.interval_ms for k in range(self.n_pulses)]

    @property
    def last_pulse_end_ms(self) -> float:
        return self.start_ms + (self.n_pulses - 1) * self.interval_ms + self.width_ms

    @property
    def stop_ms(self) -> int:
        """Window stop: middle of the gap after the last pulse (whole ms)."""
        gap = self.interval_ms - self.width_ms
        return int(math.floor(self.last_pulse_end_ms + gap / 2))

    def channel_def(self) -> DigitalPulse:
        return DigitalPulse(self.channel, start_ms=int(self.start_ms), stop_ms=self.stop_ms,
                            freq_hz=self.freq_hz, width_us=self.width_us, output=self.output)


def pulse_train_sequence(n_pulses=5, width_ms=10.0, interval_ms=100.0, start_ms=0, channel=1, output=None,
                         extra=(), max_sysclk_hz=SAFE_SYSCLK_HZ, external_trigger=False):
    """
    Sequence with one pulse train (default: 5 x 10 ms pulses, 100 ms apart, on DIG1 -> OUT1),
    plus any `extra` channel definitions (e.g. a camera clock). SYSCLK is chosen so all PWM
    frequencies share one GCLK4 prescaler and pulse widths are whole ticks.
    Returns (Sequence, PulseTrain).
    """
    train = PulseTrain(n_pulses, width_ms, interval_ms, int(start_ms), channel, output)
    channels = [train.channel_def(), *extra]
    pwm = [c for c in channels if isinstance(c, DigitalPulse)]
    freqs, widths = [c.freq_hz for c in pwm], [c.width_us for c in pwm]
    sysclk = choose_sysclk(freqs, min_width_us=min(widths), max_hz=max_sysclk_hz, widths_us=widths)
    if sysclk is None:
        raise ValueError(f"No TriggerBox system clock fits PWM frequencies {freqs} Hz with widths {widths} us "
                         f"(frequency spread too wide for the shared prescaler, or pulses too short)")
    seq = Sequence(sysclk_hz=sysclk, external_trigger=external_trigger)
    seq.add(*channels)
    seq.validate()
    return seq, train


def sequence_from_outputs(outputs, max_sysclk_hz=SAFE_SYSCLK_HZ, external_trigger=False):
    """
    One Sequence from per-output settings (the GUI's DO#1..8 and AO#1..4 rows).

    Digital entries (kind "digital" or no kind): keys
        channel (1..8; DOn -> DIGn -> OUTn), start_ms, width_ms, interval_ms, n_pulses
    n_pulses == 0  -> output not used
    n_pulses == 1  -> single level pulse (DIGTRIG), any channel 1-8, whole-ms start and width
    n_pulses >= 2  -> pulse train (DIG PWM), channels 1-5, interval = onset-to-onset = 1000/N ms

    Analog entries (kind "analog"): keys
        channel (1..4), start_ms, duration_ms, v_start, v_stop (0-5 V)
    duration_ms == 0 -> output not used
    v_start == v_stop -> fixed level (ANA); otherwise a linear ramp (ANARAMP) over the duration.
    The output returns to 0 V after start + duration. Whole-ms start and duration, 1 mV resolution.

    Returns (Sequence, summary list): digital {channel, onsets_ms, width_ms, end_ms};
    analog {kind: "analog", channel, start_ms, end_ms, v_start, v_stop}.
    """
    channels, summary = [], []
    for o in outputs:
        if o.get("kind") == "analog":
            ana = _analog_channel(o)
            if ana is not None:
                channels.append(ana)
                summary.append({"kind": "analog", "channel": ana.channel, "start_ms": ana.start_ms,
                                "end_ms": ana.stop_ms, "v_start": float(o["v_start"]), "v_stop": float(o["v_stop"])})
            continue
        ch, n = int(o["channel"]), int(o["n_pulses"])
        start, width, interval = float(o["start_ms"]), float(o["width_ms"]), float(o["interval_ms"])
        if n <= 0:
            continue
        tag = f"DO#{ch}"
        if not 1 <= ch <= N_DIGITAL:
            raise ValueError(f"{tag}: channel must be 1-{N_DIGITAL}")
        if start < 0 or abs(start - round(start)) > 1e-9:
            raise ValueError(f"{tag}: start must be a whole number of ms >= 0 (got {start})")
        if width <= 0:
            raise ValueError(f"{tag}: pulse width must be > 0")
        start = int(round(start))
        if n == 1:
            if abs(width - round(width)) > 1e-9:
                raise ValueError(f"{tag}: a single pulse is a level output with 1 ms resolution; "
                                 f"use a whole-ms width (got {width})")
            channels.append(DigitalTrigger(ch, start_ms=start, stop_ms=start + int(round(width)), output=ch))
            summary.append({"channel": ch, "onsets_ms": [start], "width_ms": width, "end_ms": start + width})
        else:
            try:
                train = PulseTrain(n, width, interval, start, ch, ch)
            except ValueError as e:
                raise ValueError(f"{tag}: {e}") from None
            channels.append(train.channel_def())
            summary.append({"channel": ch, "onsets_ms": train.onsets_ms, "width_ms": width,
                            "end_ms": train.last_pulse_end_ms})
    if not channels:
        raise ValueError("No output enabled (set N > 0 for a DO or a duration > 0 for an AO)")
    pwm = [c for c in channels if isinstance(c, DigitalPulse)]
    if pwm:
        freqs, widths = [c.freq_hz for c in pwm], [c.width_us for c in pwm]
        sysclk = choose_sysclk(freqs, min_width_us=min(widths), max_hz=max_sysclk_hz, widths_us=widths)
        if sysclk is None:
            raise ValueError(f"No TriggerBox system clock fits pulse frequencies {freqs} Hz together "
                             f"(spread too wide for the shared prescaler, or pulses too short)")
    else:
        sysclk = 1000  # level outputs / analog only: 1 ms ticks match the ms resolution
    seq = Sequence(sysclk_hz=sysclk, external_trigger=external_trigger)
    seq.add(*channels)
    seq.validate()
    return seq, summary


def _analog_channel(o):
    """AnalogFixed / AnalogRamp from one AO row, or None if the row is off (duration 0)."""
    ch = int(o["channel"])
    tag = f"AO#{ch}"
    start, dur = float(o["start_ms"]), float(o["duration_ms"])
    if dur <= 0:
        return None
    if not 1 <= ch <= N_ANALOG:
        raise ValueError(f"{tag}: channel must be 1-{N_ANALOG}")
    for name, v in (("start", start), ("duration", dur)):
        if v < 0 or abs(v - round(v)) > 1e-9:
            raise ValueError(f"{tag}: {name} must be a whole number of ms >= 0 (got {v:g})")
    mv = []
    for name in ("v_start", "v_stop"):
        v = float(o[name])
        if not 0 <= v * 1000 <= MAX_MV:
            raise ValueError(f"{tag}: {name.replace('_', ' ')} {v:g} V outside 0-{MAX_MV / 1000:g} V")
        mv.append(int(round(v * 1000)))
    start, stop = int(round(start)), int(round(start + dur))
    if mv[0] == mv[1]:
        return AnalogFixed(ch, start_ms=start, stop_ms=stop, mv=mv[0])
    return AnalogRamp(ch, start_ms=start, stop_ms=stop, start_mv=mv[0], stop_mv=mv[1])


def describe_outputs(summary) -> str:
    """One-line description of a sequence_from_outputs summary (for the GUI / log)."""
    parts = []
    for s in summary:
        if s.get("kind") == "analog":
            v = (f"{s['v_start']:g} V" if s["v_start"] == s["v_stop"]
                 else f"{s['v_start']:g} -> {s['v_stop']:g} V")
            parts.append(f"AO#{s['channel']}: {v}, {s['start_ms']:g}-{s['end_ms']:g} ms")
        else:
            on = s["onsets_ms"]
            parts.append(f"DO#{s['channel']}: {len(on)} x {s['width_ms']:g} ms"
                         + (f" every {on[1] - on[0]:g} ms" if len(on) > 1 else "") + f" from {on[0]:g} ms")
    return "; ".join(parts)


class TriggerBox:
    def __init__(self, port: str = "COM7", baud: int = 115200, timeout: float = 1.0, quiet: float = 0.05,
                 raise_on_error: bool = True, logg=None):
        """
        timeout: max wait for the first reply byte (s)
        quiet:   reply is considered complete after this idle gap (s)
        """
        self.logg = logg or logger.setup_logging()
        self.timeout = timeout
        self.quiet = quiet
        self.raise_on_error = raise_on_error
        self.last_sequence: Optional[Sequence] = None
        self._lock = threading.RLock()  # one command/reply exchange at a time across threads
        self._ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0)
        time.sleep(0.1)
        self._ser.reset_input_buffer()

    # ---- context / lifecycle -------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    @staticmethod
    def list_ports() -> list[str]:
        if list_ports is None:
            return []
        return [f"{p.device}  {p.description}" for p in list_ports.comports()]

    # ---- low level -------------------------------------------------------
    def _read_reply(self, timeout: float) -> str:
        buf = bytearray()
        t0 = last = time.monotonic()
        got = False
        while True:
            n = self._ser.in_waiting
            now = time.monotonic()
            if n:
                buf += self._ser.read(n)
                last, got = now, True
            elif got and now - last >= self.quiet:
                break
            elif not got and now - t0 >= timeout:
                break
            else:
                time.sleep(0.001)
        text = buf.decode("ascii", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def send(self, cmd: str, timeout: Optional[float] = None) -> str:
        """Send one command, return the reply text. Raises TriggerBoxError on 'ERR N'."""
        cmd = cmd.strip()
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write(cmd.encode("ascii") + b"\r")
            reply = self._read_reply(self.timeout if timeout is None else timeout)
        self.logg.info(f">> {cmd}\n   {reply.replace(chr(10), chr(10) + '   ') or '[no reply]'}")
        m = re.search(r"ERR\s*(\d+)\s*:?\s*(.*)", reply)
        if m and self.raise_on_error:
            raise TriggerBoxError(int(m.group(1)), m.group(2).strip(), cmd)
        if "warn" in reply.lower():
            warnings.warn(f"{cmd}: {reply}")
        return reply

    # ---- low-latency path (timed experiments) -----------------------------
    def _check_reply(self, cmd: str, reply: str) -> str:
        m = re.search(r"ERR\s*(\d+)\s*:?\s*(.*)", reply)
        if m and self.raise_on_error:
            raise TriggerBoxError(int(m.group(1)), m.group(2).strip(), cmd)
        return reply

    def write_timed(self, cmd: str) -> tuple[float, float]:
        """
        Write one command without waiting for the reply; returns perf_counter (before, after) the write.
        The reply stays in the input buffer: collect it with read_reply_fast(). Hold self._lock around
        the pair if other threads could talk to the box.
        """
        data = cmd.strip().encode("ascii") + b"\r"
        self._ser.reset_input_buffer()
        t0 = time.perf_counter()
        self._ser.write(data)
        t1 = time.perf_counter()
        return t0, t1

    def read_reply_fast(self, cmd: str, timeout: float = 1.0, quiet: float = 0.003) -> str:
        """
        Reply to a command sent with write_timed(): complete once a line ending has arrived and the
        line has been idle for `quiet` s (instead of the normal 50 ms idle gap). Raises on 'ERR N'.
        """
        buf = bytearray()
        t0 = last = time.perf_counter()
        while True:
            n = self._ser.in_waiting
            now = time.perf_counter()
            if n:
                buf += self._ser.read(n)
                last = now
            elif buf and (b"\r" in buf or b"\n" in buf) and now - last >= quiet:
                break
            elif now - t0 >= timeout:
                break
            else:
                time.sleep(0.0002)
        reply = buf.decode("ascii", errors="replace").replace("\r\n", "\n").replace("\r", "\n").strip()
        return self._check_reply(cmd.strip(), reply)

    def command_fast(self, cmd: str, timeout: float = 1.0, quiet: float = 0.003) -> tuple[float, float, str]:
        """write_timed + read_reply_fast under the lock: (t_write_start, t_write_end, reply). Not logged."""
        with self._lock:
            t0, t1 = self.write_timed(cmd)
            return t0, t1, self.read_reply_fast(cmd, timeout, quiet)

    def status_fast(self) -> tuple[str, int]:
        _, _, raw = self.command_fast("STATUS", timeout=0.5, quiet=0.0)  # one line: done at its line end
        m = re.search(r"STATUS:\s*([^,\r\n]+)\s*,\s*TICK:\s*(\d+)", raw, re.I)
        return (m.group(1).strip(), int(m.group(2))) if m else (raw, -1)

    # ---- system commands -------------------------------------------------
    def reset(self) -> str:
        r = self.send("RESET")
        time.sleep(0.1)
        return r

    def set_sysclk(self, hz: int) -> str:
        return self.send(f"SYSCLK:16:{hz}")

    def set_trigger_mode(self, external: bool) -> str:
        return self.send(f"TRIG:8:{int(external)}")

    def set_max_time(self, ms: int) -> str:
        return self.send(f"MAXTIME:32:{ms}")

    def set_com2_baud(self, baud: int) -> str:
        # Format not given in spec v2.0; assumed to follow the uint32 convention.
        return self.send(f"COM2TXBAUD:32:{baud}")

    def arm(self) -> dict:
        """ARM and return parsed feedback (raw text in 'raw')."""
        raw = self.send("ARM", timeout=2.0)
        info = {"raw": raw}
        m = re.search(r"prescaler\s*=\s*(\d+)", raw, re.I)
        if m:
            info["gclk4_prescaler"] = int(m.group(1))
        m = re.search(r"System Clock:\s*([\d.,]+)\s*Hz", raw, re.I)
        if m:
            info["sysclk_hz"] = float(m.group(1).replace(",", "."))
        info["dig"] = {int(n): (float(fq.replace(",", ".")), rest.strip())
                       for n, fq, rest in re.findall(r"DIG(\d)\s*:\s*([\d.,]+)\s*Hz,?\s*([^\r\n]*)", raw)}
        return info

    def run(self) -> str:
        return self.send("RUN")

    def stop(self) -> str:
        return self.send("STOP")

    # ---- status ------------------------------------------------------------
    def status(self) -> tuple[str, int]:
        raw = self.send("STATUS")
        m = re.search(r"STATUS:\s*([^,\r\n]+)\s*,\s*TICK:\s*(\d+)", raw, re.I)
        return (m.group(1).strip(), int(m.group(2))) if m else (raw, -1)

    def _kv(self, raw: str) -> dict:
        return {k: v for k, v in re.findall(r"(\w+)=([^,\s]+)", raw)}

    def dig_status(self, n: int) -> dict:
        return self._kv(self.send(f"DIGSTAT{n}"))

    def ana_status(self, n: int) -> dict:
        return self._kv(self.send(f"ANASTAT{n}"))

    # ---- high level --------------------------------------------------------
    def upload(self, seq: Sequence, reset: bool = True, validate: bool = True):
        """RESET (optional) and send the full configuration. Does not ARM."""
        if validate:
            seq.validate()
        if reset:
            self.reset()
        for cmd in seq.commands():
            self.send(cmd)
        self.last_sequence = seq

    def wait_until_done(self, timeout: Optional[float] = None, poll: float = 0.1) -> tuple[str, int]:
        """Poll STATUS until the sequence has finished (auto-stop)."""
        seq = self.last_sequence
        expected = seq.duration_ms / 1000 if seq else 0.0
        if timeout is None:
            timeout = float("inf") if (seq and seq.external_trigger) else expected + 2.0
        t0, seen_running = time.monotonic(), False
        while True:
            state, tick = self.status()
            running = "RUN" in state.upper()
            seen_running |= running
            elapsed = time.monotonic() - t0
            if not running and (seen_running or elapsed > expected):
                break
            if elapsed > timeout:
                raise TimeoutError(f"Sequence not finished after {timeout:.1f} s (state {state})")
            time.sleep(poll)
        self.logg.info(f"   done: STATUS {state}, TICK {tick}")
        return state, tick

    def execute(self, seq: Sequence, wait: bool = True) -> dict:
        """RESET, configure, ARM, RUN (software trigger) and optionally wait."""
        self.upload(seq)
        info = self.arm()
        if seq.external_trigger:
            self.logg.info("   armed, waiting for external trigger on TRIG input")
        else:
            self.run()
        if wait:
            self.wait_until_done()
        return info

    def fire_outputs(self, outputs, wait=True) -> list:
        """Build a sequence from per-output settings (see sequence_from_outputs), upload, ARM, RUN."""
        seq, summary = sequence_from_outputs(outputs)
        self.execute(seq, wait=wait)
        return summary

    def run_pulse_train(self, n_pulses=5, width_ms=10.0, interval_ms=100.0, start_ms=0, channel=1,
                        output=None, wait=True) -> PulseTrain:
        """Upload, ARM and RUN a single pulse train (default 5 x 10 ms, 100 ms apart, DIG1 -> OUT1)."""
        seq, train = pulse_train_sequence(n_pulses, width_ms, interval_ms, start_ms, channel, output)
        self.execute(seq, wait=wait)
        return train

    def repl(self):
        print("Triggerbox terminal. Empty line or 'exit' quits.")
        raise_, self.raise_on_error = self.raise_on_error, False
        try:
            while True:
                cmd = input("tb> ").strip()
                if cmd.lower() in ("", "exit", "quit"):
                    break
                self.send(cmd)
        finally:
            self.raise_on_error = raise_


if __name__ == '__main__':
    print(TriggerBox.list_ports())

    # 5 x 10 ms pulses, 100 ms apart, on DIG1 -> OUT1: preview, then run
    seq, train = pulse_train_sequence(n_pulses=5, width_ms=10, interval_ms=100)
    print(seq.commands(), train.onsets_ms)
    seq.plot()

    tb = TriggerBox()
    tb.run_pulse_train(n_pulses=5, width_ms=10, interval_ms=100)

    seq = Sequence(sysclk_hz=10_000, external_trigger=False)
    seq.add(
        DigitalPulse(1, start_ms=10, stop_ms=95, freq_hz=100, width_us=200, output=1),
        DigitalTrigger(3, start_ms=0, stop_ms=101, output=3),  # HIGH from start to stop
        AnalogFixed(1, 0, 500, mv=2500),
        AnalogRamp(2, 0, 1000, start_mv=0, stop_mv=5000),
    )
    seq.validate()  # raises ValueError on errors, emits warnings for risky settings
    seq.commands()  # the command strings that will be sent
    seq.plot()  # expected waveforms, no hardware needed

    tb.upload(seq)  # RESET + config (validates first)
    info = tb.arm()  # dict: raw, gclk4_prescaler, sysclk_hz, dig{n: (Hz, text)}
    tb.run()
    tb.wait_until_done()  # polls STATUS; timeout defaults to duration + 2 s
    tb.stop()  # stop early; config kept, run() again without re-ARM

    tb.close()
