# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_uniform_phase(size=(1536, 2048), ph=0, typ=np.uint8):
    if ph:
        return 255 * np.ones(size, dtype=typ)
    else:
        return np.zeros(size, dtype=typ)


def generate_binary_phase_1bit(size=(2048, 1536), period=(8, 0), phase=(0, 0), value=255, typ=np.uint8):
    width, height = size
    period_x, period_y = period
    offset_x, offset_y = phase
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    xx += offset_x
    yy += offset_y
    if period_x > 0 and period_y > 0:
        return np.where(((xx % period_x) < (period_x // 2)) ^ ((yy % period_y) < (period_y // 2)), value, 0).astype(typ)
    if period_x > 0 and period_y == 0:
        return np.where(((xx % period_x) < (period_x // 2)), value, 0).astype(typ)
    if period_x == 0 and period_y > 0:
        return np.where(((yy % period_y) < (period_y // 2)), value, 0).astype(typ)
    return None


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


def simulate_binary_phase_pattern(size=(1024, 1024), period=(8, 0), phase=(0, 0), value=1, cutoff=100):
    width, height = size
    period_x, period_y = period
    offset_x, offset_y = phase
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    xx += offset_x
    yy += offset_y
    if period_x > 0 and period_y > 0:
        pattern = np.where(((xx % period_x) < (period_x // 2)) ^ ((yy % period_y) < (period_y // 2)), value, 0)
    elif period_x > 0 and period_y == 0:
        pattern = np.where(((xx % period_x) < (period_x // 2)), value, 0)
    elif period_x == 0 and period_y > 0:
        pattern = np.where(((yy % period_y) < (period_y // 2)), value, 0)
    else:
        pattern = np.ones(size)
    pattern_field = 1.0 * np.exp(1j * np.pi * pattern)
    pupil_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pattern_field)))
    pupil_maks = ((xx - width // 2) ** 2 + (yy - height // 2) ** 2) <= cutoff ** 2
    pupil_filtered = pupil_maks * pupil_field
    focal_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil_filtered)))
    focal_intensity = np.abs(focal_field) ** 2
    return np.abs(pupil_filtered), focal_intensity


def simulate_phase_pattern(N=1024, dx=0.1e-6, wavelength=488e-9, NA=1.3,
                           grating_period=1.20001e-6, duty_cycle=0.5, phase_depth=np.pi, orientation_deg=0, grating_shift=0.0,
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
