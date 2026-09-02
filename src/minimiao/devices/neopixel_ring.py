# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np

from minimiao import logger


class NeoPixel:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.sample_rate = 8e6
        self.time_res = 1 / self.sample_rate
        self.bit_time = 1.25e-6  # s
        self.bit_sample = int(self.bit_time * self.sample_rate)
        self.bit_0_high = 2
        self.bit_0_low = 8
        self.bit_1_high = 7
        self.bit_1_low = 3
        self.reset_time = 80e-6  # s
        self.reset_sample = int(self.reset_time * self.sample_rate)

    def close(self):
        pass

    def pulse_wave(self, bit_type=0):
        """Return time and waveform for one bit ('0' or '1')."""
        if bit_type:
            bit_high = self.bit_1_high
        else:
            bit_high = self.bit_0_high
        t = np.linspace(0, self.bit_time * 1e6, self.bit_sample, endpoint=False)
        x = np.zeros(self.bit_sample, dtype=np.uint8)
        x[:bit_high] = 1
        return t, x

    def byte_wave(self, byte_val):
        """Function to build waveform for a byte (MSB first)"""
        bits = [(byte_val >> (7 - i)) & 1 for i in range(8)]  # MSB first
        t_byte, x_byte = [], []
        for i, b in enumerate(bits):
            t, x = self.pulse_wave(b)
            if len(t_byte) == 0:
                t_byte, x_byte = t, x
            else:
                t_byte = np.concatenate([t_byte, t + t_byte[-1] + self.time_res])
                x_byte = np.concatenate([x_byte, x])
        return t_byte, x_byte

    def color_wave(self, g, r, b, w):
        _, y_g = self.byte_wave(g)
        _, y_r = self.byte_wave(r)
        _, y_b = self.byte_wave(b)
        _, y_w = self.byte_wave(w)
        y_all = np.concatenate([y_g, y_r, y_b, y_w])
        t_all = np.arange(len(y_all)) * self.time_res * 1e6
        return t_all, y_all

    def led_ring(self, colors=None):
        if colors is None:
            colors = [[0, 0, 0, 0]] * 24
        y_all = []
        for g, r, b, w in colors:
            _, y_led = self.color_wave(g, r, b, w)
            y_all = np.concatenate([y_all, y_led]) if len(y_all) else y_led
        return y_all

    def single_led(self, num, n, color):
        c = [[0, 0, 0, 0]] * num
        h = int(num / 2)
        c[n] = color
        x = self.led_ring(colors=c)
        return x

    def half_ring(self, num, start, h, color):
        c = [[0, 0, 0, 0]] * num
        for i in range(h):
            idx = (start + i) % num
            c[idx] = color
        x = self.led_ring(colors=c)
        return x

    def dpc_sequences(self, c, amp, expo, rdt):
        hd = np.zeros(int(expo * self.sample_rate))
        rd = np.zeros(int(rdt * self.sample_rate))
        off = self.led_ring([[0, 0, 0, 0]] * 24)
        color = [0, 0, 0, 0]
        color[c] = amp
        # xhp = self.single_led(24, 0, color)
        # xhn = self.single_led(24, 12, color)
        # yhp = self.single_led(24, 6, color)
        # yhn = self.single_led(24, 18, color)
        xhp = self.half_ring(24, 0, 10, color)
        xhn = self.half_ring(24, 12, 10, color)
        yhp = self.half_ring(24, 6, 10, color)
        yhn = self.half_ring(24, 18, 10, color)
        led_seq = np.hstack((xhp, hd, off, rd, xhn, hd, off, rd, yhp, hd, off, rd, yhn, hd, off, rd))
        stby = np.zeros(xhp.shape)
        expo = np.ones(int(expo * self.sample_rate))
        ot = np.zeros(off.shape)
        cam_seq = np.hstack((stby, expo, ot, rd, stby, expo, ot, rd, stby, expo, ot, rd, stby, expo, ot, rd))
        return np.vstack((led_seq, cam_seq))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    p = NeoPixel()

    ts, xs = p.byte_wave(0xFF)
    plt.figure(figsize=(12, 3))
    plt.step(ts, xs, where='post')
    plt.xlabel("Time (µs)")
    plt.ylabel("Signal level")
    plt.title("NeoPixel Data Pulse for Bright Green (GRB = FF0000)")
    plt.grid(True)
    plt.show()

    ts, xs = p.color_wave(0xFF, 0x00, 0x00)
    plt.figure(figsize=(12, 3))
    plt.step(ts, xs, where='post')
    plt.xlabel("Time (µs)")
    plt.ylabel("Signal level")
    plt.title("NeoPixel Data Pulse for Bright Green (GRB = FF0000)")
    plt.grid(True)
    plt.show()

    ts, xs = p.led_ring()
    plt.figure(figsize=(12, 3))
    plt.step(ts, xs, where='post')
    plt.xlabel("Time (µs)")
    plt.ylabel("Signal level")
    plt.title("NeoPixel Ring 16: Full Data Frame (16 × 24 bits + Reset)")
    plt.grid(True)
    plt.show()
