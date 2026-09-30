# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from math import gcd

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_uniform_phase(size=(1536, 2048), ph=0, typ=np.uint8):
    if ph:
        return 255 * np.ones(size, dtype=typ)
    else:
        return np.zeros(size, dtype=typ)


def generate_binary_phase_1bit(size=(2048, 1536), period=(8, 0), phase=(0, 0), duty=0.5, value=255, typ=np.uint8):
    width, height = size
    period_x, period_y = period
    offset_x, offset_y = phase
    duty_x, duty_y = duty if np.iterable(duty) else (duty, duty)

    def on_width(period, duty):
        return int(np.clip(round(duty * period), 1, period - 1))

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = xx + offset_x
    yy = yy + offset_y

    mask_x = (xx % period_x) < on_width(period_x, duty_x) if period_x > 0 else None
    mask_y = (yy % period_y) < on_width(period_y, duty_y) if period_y > 0 else None

    if mask_x is not None and mask_y is not None:
        mask = mask_x ^ mask_y
    elif mask_x is not None:
        mask = mask_x
    elif mask_y is not None:
        mask = mask_y
    else:
        return None
    return np.where(mask, value, 0).astype(typ)


def generate_binary_phase_8bit(bit_sequences):
    bit_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    width, height = bit_sequences[0].shape
    patterns = np.zeros((8, width, height), dtype=np.uint8)
    pattern = np.zeros((width, height), dtype=np.uint8)
    for i, bn in enumerate(bit_indices):
        patterns[bn] = bit_sequences[i]
    for i in range(8):
        pattern += patterns[i] * (2 ** i)
    return pattern


def save_to_bmp(data, svd, fn, bt=1):
    img = Image.fromarray(data, mode='L')
    if bt:
        img = img.convert('1', dither=Image.NONE)
        img.save(svd + fn + r"_1bit.bmp", format='BMP')
    else:
        img.save(svd + fn + r"_8bit.bmp", format='BMP')


def generate_tilted_binary_phase(size=(2048, 1536), vec=(7, 4), period=12, step=0, nsteps=3,
                                 duty=0.5, value=255, typ=np.uint8):
    a, b = vec
    if b <= 0:
        raise ValueError("b must be > 0")
    L = b * period
    if (L * step) % nsteps:
        raise ValueError(f"b*period ({L}) not divisible by nsteps ({nsteps})")
    width, height = size
    n_on = int(np.clip(round(duty * L), 1, L - 1))
    yy, xx = np.mgrid[:height, :width]
    m = np.mod(b * xx - a * yy + step * L // nsteps, L)   # shift applied here only
    return np.where(m < n_on, value, 0).astype(typ)


def pattern_geometry(vec, period):
    """
    Line spacing (px) and grating-vector angle (deg, as displayed with y down).
    Example usage:
        s, ang = pattern_geometry((15, 26), 12)
        print(f"spacing={s:.3f} px, angle={ang:.2f} deg")
    """
    a, b = vec
    spacing = period * b / np.hypot(a, b)
    angle = np.degrees(np.arctan2(a, b))
    return spacing, angle


def find_lattice(target_spacing, angle_deg, nsteps=3, max_period=60, max_b=12, n_best=5):
    """
    Search integer (a, b, period) matching a target spacing and angle (Supp. Fig. 2 constraints).
    Example usage:
        for c in find_lattice(15, 30, nsteps=5, max_period=15, max_b=20, n_best=5):
            print(" ", c)
    """
    out = []
    t = np.tan(np.radians(angle_deg))
    for period in range(1, max_period + 1):
        for b in range(1, max_b + 1):
            if (b * period) % nsteps:
                continue
            a = int(round(b * t))
            if gcd(abs(a), b) != 1 and a != 0:
                continue
            if a == 0 and b != 1:
                continue
            s, ang = pattern_geometry((a, b), period)
            out.append((abs(s / target_spacing - 1), abs(ang - angle_deg), (a, b), period, s, ang))
    out.sort(key=lambda r: (round(r[0] + np.radians(r[1]), 3), r[3]))
    return [dict(vec=r[2], period=r[3], spacing=r[4], angle=r[5]) for r in out[:n_best]]


def diffraction_orders(vec, period, duty=0.5, rel_threshold=1e-3):
    """
    Exact far-field orders of the 0/pi pattern from one lattice supercell (Supp. Fig. 4).
    Returns array rows (fx, fy, power) in cycles/pixel, sorted by power; power normalized to total.
    Example usage:
        print("\nStrongest orders, +60 pattern (fx, fy [cyc/px], power):")
        print(np.round(diffraction_orders((15, 26), 12)[:9], 4))
    """
    a, b = vec
    L = b * period
    ty = L // gcd(abs(a), L)  # vertical period of the pattern
    tile = generate_tilted_binary_phase((period, ty), vec, period, 0, 1, duty, 1, np.int8)
    field = 1.0 - 2.0 * tile  # 0 / pi phase
    F = np.fft.fft2(field) / field.size
    P = np.abs(F) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(ty), np.fft.fftfreq(period), indexing="ij")
    keep = P > rel_threshold * P.max()
    rows = np.column_stack([fx[keep], fy[keep], P[keep]])
    return rows[np.argsort(-rows[:, 2])]


def pupil_map(patterns, beam_radius=0.9, duty=0.5, rel_threshold=1e-3):
    """
    Order positions in normalized pupil coordinates (pupil radius = 1) when the
    +-1 orders are placed at `beam_radius`. Assumes all patterns share ~the same spacing.
    Returns list of (label, rho_x, rho_y, power, is_main).
    """
    spacings = [pattern_geometry(v, p)[0] for v, p in patterns]
    f1 = 1.0 / np.mean(spacings)  # main-order frequency, cycles/px
    res = []
    for (v, p) in patterns:
        for fx, fy, pw in diffraction_orders(v, p, duty, rel_threshold):
            rx, ry = fx / f1 * beam_radius, fy / f1 * beam_radius
            main = np.isclose(np.hypot(fx, fy), 1 / pattern_geometry(v, p)[0]) and pw > 0.1
            if np.hypot(rx, ry) <= 1.5:
                res.append((f"{v},{p}", rx, ry, pw, main))
    return res


def generate_hex_binary_phase(size=(2048, 1536), vec=(17, 30), period=14, shift=(0, 0), nsteps=12,
                              duty=0.5, value=255, typ=np.uint8):
    """
    Binary hexagonal pattern from three gratings:
        g1: vec=( a, b)  -> m1 = b*x - a*y
        g2: vec=(-a, b)  -> m2 = b*x + a*y
        g3: horizontal   -> m3 = m2 - m1 = 2a*y   (spacing L/(2a))
    with L = b*period. g3 = g2 - g1 exactly, so the three k-vectors close a
    triangle and the pattern is exactly periodic on an L x L cell.
    Pixels are on where cos(phi1) + cos(phi2) + cos(phi3) > threshold,
    with the threshold set to give the requested on-fraction (duty).

    shift=(s1, s2): phase offsets of g1 and g2 in units of 2*pi/nsteps;
    g3 then shifts by s2 - s1.
    """
    a, b = vec
    L = b * period
    s1, s2 = shift
    if (L * s1) % nsteps or (L * s2) % nsteps:
        raise ValueError(f"b*period ({L}) not divisible by nsteps ({nsteps})")
    d1, d2 = s1 * L // nsteps, s2 * L // nsteps

    def field(xx, yy):
        m1 = np.mod(b * xx - a * yy + d1, L)
        m2 = np.mod(b * xx + a * yy + d2, L)
        m3 = np.mod(m2 - m1, L)
        w = 2 * np.pi / L
        return np.cos(w * m1) + np.cos(w * m2) + np.cos(w * m3)

    yy, xx = np.mgrid[:L, :L]                      # one exact period cell
    thr = np.quantile(field(xx, yy), 1 - duty)

    width, height = size
    yy, xx = np.mgrid[:height, :width]
    return np.where(field(xx, yy) > thr, value, 0).astype(typ)


def hex_geometry(vec, period):
    a, b = vec
    L = b * period
    ang = np.degrees(np.arctan2(a, b))
    return dict(spacing_tilted=L / np.hypot(a, b),
                spacing_horizontal=L / (2 * abs(a)),
                angles=(ang, -ang, 90.0))


def generate_binary_phase_dots(size=(2048, 1536), period=(8, 8), phase=(0, 0),
                               geometry='checker', threshold=None,
                               value=255, typ=np.uint8):
    """Binary (0 / value) phase pattern that synthesizes a dot array at the
    sample, given an order-selection mask in the intermediate pupil.

    geometry : 'checker' | 'square' | 'hex'
    period   : (period_x, period_y) in SLM pixels. 'hex' uses period_x only.
    phase    : (offset_x, offset_y) in pixels; shifts the pattern rigidly.
    threshold: binarization level for the cosine modes. None -> median,
               which equalizes the 0/pi areas and nulls the zero order.
    """
    width, height = size
    period_x, period_y = period
    offset_x, offset_y = phase

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = xx + offset_x
    yy = yy + offset_y

    if geometry == 'checker':
        if period_x > 0 and period_y > 0:
            mask = ((xx % period_x) < (period_x // 2)) ^ ((yy % period_y) < (period_y // 2))
        elif period_x > 0:
            mask = (xx % period_x) < (period_x // 2)
        elif period_y > 0:
            mask = (yy % period_y) < (period_y // 2)
        else:
            return None
        return np.where(mask, value, 0).astype(typ)

    if geometry == 'square':
        if period_x <= 0 or period_y <= 0:
            return None
        field = np.cos(2 * np.pi * xx / period_x) + np.cos(2 * np.pi * yy / period_y)
    elif geometry == 'hex':
        if period_x <= 0:
            return None
        field = np.zeros((height, width))
        for angle in np.deg2rad((0.0, 60.0, 120.0)):
            field += np.cos(2 * np.pi * (np.cos(angle) * xx +
                                         np.sin(angle) * yy) / period_x)
    else:
        raise ValueError(f'unknown geometry: {geometry!r}')

    level = np.median(field) if threshold is None else threshold
    return np.where(field < level, value, 0).astype(typ)


def generate_fresnel_lens_pattern(size=(1272, 1024), ps=12.5e-6, wl=488e-9,
                                  cnt=((0, 4e-3), (0, -4e-3)), fl=(0.25, 0.25), bd=10e-3):
    slm_width, slm_height = size
    pixel_pitch = ps
    wavelength = wl
    centers = cnt
    focal_lengths = fl
    mask_diameter = bd
    mask_radius = (mask_diameter / 2) / pixel_pitch
    x = np.arange(slm_width)
    y = np.arange(slm_height)
    xv, yv = np.meshgrid(x, y)
    center_x_px = slm_width // 2
    center_y_px = slm_height // 2
    r_mask = np.sqrt((xv - center_x_px) ** 2 + (yv - center_y_px) ** 2)
    mask = (r_mask <= mask_radius).astype(float)
    if isinstance(focal_lengths, (float, int)):
        focal_lengths = [focal_lengths] * len(centers)
    elif len(focal_lengths) != len(centers):
        raise ValueError("focal_lengths must match the length of centers.")
    phase_total = np.zeros_like(xv, dtype=np.float64)
    for (x_mm, y_mm), f in zip(centers, focal_lengths):
        x_px_offset = x_mm / pixel_pitch
        y_px_offset = y_mm / pixel_pitch
        cx = center_x_px + x_px_offset
        cy = center_y_px + y_px_offset
        x_m = (xv - cx) * pixel_pitch
        y_m = (yv - cy) * pixel_pitch
        r2 = x_m ** 2 + y_m ** 2
        phase = (-np.pi * r2) / (wavelength * f)
        phase_total += phase * mask
    phase_wrapped = np.mod(phase_total, 2 * np.pi)
    phase_img = np.uint8(255 * phase_wrapped / (2 * np.pi))
    return phase_img


def generate_blazed_pattern(size=(1272, 1024), ps=12.5e-6, wl=488e-9, pd=50):
    slm_width, slm_height = size
    pixel_pitch = ps
    wavelength = wl
    grating_period = pd

    d = grating_period * pixel_pitch
    sin_theta = wavelength / d
    if abs(sin_theta) > 1:
        raise ValueError("grating_period_px too small for physical steering! Increase period.")
    theta_rad = np.arcsin(sin_theta)
    theta_deg = np.degrees(theta_rad)
    print(f"Grating period: {grating_period} px, steering angle: {theta_deg:.2f} deg")

    x = np.arange(slm_width)
    blaze = 2 * np.pi * (x % grating_period) / grating_period
    phase_pattern = np.tile(blaze, (slm_height, 1))
    phase_img = np.uint8(255 * phase_pattern / (2 * np.pi))
    return phase_img


def generate_lee_hologram(size=(1272, 1024), ps=12.5e-6, wl=488e-9, ang=4):
    slm_width, slm_height = size
    pixel_pitch = ps
    wavelength = wl
    steering_angle_deg = ang
    theta_rad = np.deg2rad(steering_angle_deg)
    k = 2 * np.pi / wavelength
    carrier_period_m = wavelength / np.sin(theta_rad)  # meters
    carrier_period_px = carrier_period_m / pixel_pitch
    carrier_freq_px = 1.0 / carrier_period_px

    x = np.arange(slm_width)
    y = np.arange(slm_height)
    xv, yv = np.meshgrid(x, y)
    carrier = 2 * np.pi * carrier_freq_px * xv

    phase_pattern = np.mod(carrier, 2 * np.pi)
    return phase_pattern


def generate_split_grating(beam_num=5, spacing=32, pixel_nums=(1024, 1272), iterations=500, binary=True):
    cent_x, cent_y = pixel_nums[0] // 2, pixel_nums[1] // 2
    beam_positions = []
    offsets = np.linspace(start=-int(spacing * int(np.floor(beam_num / 2))),
                          stop=int(spacing * int(np.floor(beam_num / 2))),
                          num=beam_num, dtype=int)
    for r_off in offsets:
        for c_off in offsets:
            beam_positions.append((cent_x + r_off, cent_y + c_off))
    field = np.random.choice([1, -1], size=pixel_nums)
    target = np.zeros(pixel_nums, dtype=float)
    for pos in beam_positions:
        r, c = pos
        target[r, c] = 1.0
    for _ in range(iterations):
        far_field = np.fft.fftshift(np.fft.fft2(field))
        phase_far = np.exp(1j * np.angle(far_field))
        far_field_new = target * phase_far
        field_new = np.fft.ifft2(np.fft.ifftshift(far_field_new))
        if binary:
            field = np.where(np.real(field_new) >= 0, 1, -1)
    return field


def simulate_binary_phase_pattern(size=(1024, 1024), period=(8, 0), phase=(0, 0), duty=0.5, value=1, cutoff=100):
    width, height = size
    period_x, period_y = period
    offset_x, offset_y = phase
    duty_x, duty_y = duty if np.iterable(duty) else (duty, duty)

    def on_width(period, duty):
        return int(np.clip(round(duty * period), 1, period - 1))

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = xx + offset_x
    yy = yy + offset_y

    mask_x = (xx % period_x) < on_width(period_x, duty_x) if period_x > 0 else None
    mask_y = (yy % period_y) < on_width(period_y, duty_y) if period_y > 0 else None
    if mask_x is not None and mask_y is not None:
        mask = mask_x ^ mask_y
    elif mask_x is not None:
        mask = mask_x
    elif mask_y is not None:
        mask = mask_y
    else:
        return None
    pattern = np.where(mask, value, 0)
    pattern_field = 1.0 * np.exp(1j * np.pi * pattern)
    pupil_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pattern_field)))
    pupil_maks = ((xx - width // 2) ** 2 + (yy - height // 2) ** 2) <= cutoff ** 2
    pupil_filtered = pupil_maks * pupil_field
    focal_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil_filtered)))
    focal_intensity = np.abs(focal_field) ** 2
    return np.abs(pupil_filtered), focal_intensity


def simulate_phase_pattern(N=1024, dx=0.1e-6, wavelength=488e-9, NA=1.3,
                           grating_period=1.20001e-6, duty_cycle=0.5, phase_depth=np.pi, orientation_deg=0,
                           grating_shift=0.0,
                           order_filter_radius_factor=0.18, verbose=False):
    L = N * dx
    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation_deg)
    # Coordinate along the grating modulation direction
    U = X * np.cos(theta) + Y * np.sin(theta)
    # Binary pattern: 0 or 1
    binary_pattern = ((U + grating_shift) % grating_period) < (duty_cycle * grating_period)
    # Binary phase: 0 or pi
    phase_mask = phase_depth * binary_pattern.astype(float)
    # Complex field immediately after phase mask
    E_mask = np.exp(1j * phase_mask)
    # Fourier transform: image-conjugate plane -> pupil / spatial-frequency plane
    E_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_mask)))
    # Spatial-frequency coordinates
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    FX, FY = np.meshgrid(fx, fy)
    # Objective coherent cutoff frequency
    f_cutoff = NA / wavelength
    # Circular objective pupil
    objective_pupil = (FX ** 2 + FY ** 2) <= f_cutoff ** 2
    # First diffraction order spatial frequency
    f1 = 1.0 / grating_period
    # ±1 order positions, oriented according to grating angle
    fx1 = f1 * np.cos(theta)
    fy1 = f1 * np.sin(theta)
    # Circular mask radius around each order
    order_filter_radius = order_filter_radius_factor * f1
    # Circular masks around +1 and -1 diffraction orders
    mask_plus1 = ((FX - fx1) ** 2 + (FY - fy1) ** 2) <= order_filter_radius ** 2
    mask_minus1 = ((FX + fx1) ** 2 + (FY + fy1) ** 2) <= order_filter_radius ** 2
    order_selection_mask = mask_plus1 | mask_minus1
    # Apply both objective pupil and ±1 order selection
    E_fourier_filtered = E_fourier * objective_pupil * order_selection_mask
    # Inverse Fourier transform: selected orders -> focal plane
    E_focal = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_fourier_filtered)))
    I_focal = np.abs(E_focal) ** 2
    I_focal /= I_focal.max()
    E_fourier_pupil_only = E_fourier * objective_pupil
    E_unfiltered = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_fourier_pupil_only)))
    I_unfiltered = np.abs(E_unfiltered) ** 2
    I_unfiltered /= I_unfiltered.max()
    if verbose:
        extent_um = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]

        # ---- Figure 1: binary phase mask ----
        plt.figure(figsize=(6, 5))
        plt.imshow(phase_mask, extent=extent_um, cmap='gray', origin='lower')
        plt.colorbar(label='Phase [rad]')
        plt.xlabel('x [µm]')
        plt.ylabel('y [µm]')
        plt.title('Binary phase modulation in conjugate image plane')
        plt.tight_layout()
        plt.show()

        # ---- Figure 2: Fourier plane diffraction pattern ----
        fourier_intensity = np.abs(E_fourier) ** 2
        fourier_intensity_log = np.log10(fourier_intensity / fourier_intensity.max() + 1e-8)

        extent_freq = [fx[0] * 1e-3, fx[-1] * 1e-3, fy[0] * 1e-3, fy[-1] * 1e-3]

        plt.figure(figsize=(6, 5))
        plt.imshow(fourier_intensity_log, extent=extent_freq, cmap='gray', origin='lower')
        plt.xlabel(r'$f_x$ [mm$^{-1}$]')
        plt.ylabel(r'$f_y$ [mm$^{-1}$]')
        plt.title('Fourier plane: diffraction orders')
        plt.colorbar(label='log10 normalized intensity')
        plt.tight_layout()
        plt.show()

        # ---- Figure 3: selected ±1 diffraction orders ----
        selected_intensity = np.abs(E_fourier_filtered) ** 2
        selected_intensity_log = np.log10(selected_intensity / selected_intensity.max() + 1e-8)

        plt.figure(figsize=(6, 5))
        plt.imshow(selected_intensity_log, extent=extent_freq, cmap='gray', origin='lower')
        plt.xlabel(r'$f_x$ [mm$^{-1}$]')
        plt.ylabel(r'$f_y$ [mm$^{-1}$]')
        plt.title('Fourier plane after selecting ±1 orders')
        plt.colorbar(label='log10 normalized intensity')
        plt.tight_layout()
        plt.show()

        # ---- Figure 4: focal-plane intensity without and with filtering ----
        plt.figure(figsize=(6, 5))
        plt.imshow(I_unfiltered, extent=extent_um, cmap='gray', origin='lower')
        plt.xlabel('x [µm]')
        plt.ylabel('y [µm]')
        plt.title('Focal plane intensity: pupil only')
        plt.colorbar(label='Normalized intensity')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 5))
        plt.imshow(I_focal, extent=extent_um, cmap='gray', origin='lower')
        plt.xlabel('x [µm]')
        plt.ylabel('y [µm]')
        plt.title('Focal plane intensity: ±1 orders only')
        plt.colorbar(label='Normalized intensity')
        plt.tight_layout()
        plt.show()

        # ---- Figure 5: central line profile of sinusoidal illumination ----
        center_line = I_focal[N // 2, :]

        plt.figure(figsize=(8, 4))
        plt.plot(x * 1e6, center_line, linewidth=2)
        plt.xlabel('x [µm]')
        plt.ylabel('Normalized intensity')
        plt.title('Central line profile of generated sinusoidal pattern')
        plt.xlim(-30, 30)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return phase_mask, np.abs(E_fourier) ** 2, np.abs(E_fourier_filtered) ** 2, I_unfiltered, I_focal
