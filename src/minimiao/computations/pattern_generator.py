# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


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
    return np.where(((xx % period_x) < (period_x // 2)) ^ ((yy % period_y) < (period_y // 2)), value, 0).astype(typ)


def generate_binary_phase_8bit(bit_indices, bit_sequences):
    if len(bit_sequences) != len(bit_indices):
        raise Exception("Error: bit index and bit sequence length does not match")
    width, height = bit_sequences[0].shape
    patterns = np.zeros((8, width, height))
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
