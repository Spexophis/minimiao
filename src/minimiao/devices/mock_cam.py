# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import run_threads
import numpy as np


class MockCamera:
    def __init__(self, shape=(2048, 2048)):
        self.shape = shape
        self.buffer_size = 4
        self.data = None
        self.acq_thread = None

    def close(self):
        pass

    def prepare_live(self):
        self.data = run_threads.CameraDataList(self.buffer_size)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)

    def start_live(self):
        self.acq_thread.start()

    def stop_live(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        self.data = None

    def prepare_acquisition(self, n):
        self.data = run_threads.CameraDataList(n)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)

    def start_acquisition(self):
        self.acq_thread.start()

    def stop_acquisition(self):
        self.acq_thread.stop()
        self.acq_thread = None

    def get_images(self):
        self.data.add_element([np.random.randint(0, 2**14, size=self.shape, dtype=np.uint16)])

    def get_last_image(self):
        return np.random.randint(0, 2**14, size=self.shape, dtype=np.uint16)
