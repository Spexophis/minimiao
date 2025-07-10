import time

import numpy as np


class SimulatedCamera:
    def __init__(self, shape=(2048, 2048)):
        self.shape = shape

    def get_frame(self):
        t = time.time()
        img = np.random.normal(80, 5, self.shape).astype(np.float32)
        x0 = int((np.sin(t) + 1) / 2 * (self.shape[0] - 40) + 20)
        y0 = int((np.cos(t) + 1) / 2 * (self.shape[1] - 40) + 20)
        img[x0 - 8:x0 + 8, y0 - 8:y0 + 8] += 100
        return np.clip(img, 0, 255).astype(np.uint8)
