# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from collections import deque

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression


class FocusLocker:

    def __init__(self, pid_p=(0.5, 0.5, 0.0)):
        self.threshold = None
        self.cm_ref = None
        self.model = LinearRegression()
        kp, ki, kd = pid_p
        self.pid = PID(P=kp, I=ki, D=kd)
        self.ctd = CtrlData(128)

    def update_pid(self, pid_p=(0.8, 0.6, 0.0)):
        if pid_p is not None:
            nkp, nki, nkd = pid_p
            self.pid.Kp = nkp
            self.pid.Ki = nki
            self.pid.Kd = nkd

    def initiate(self, zp):
        self.ctd.reset()
        self.ctd.add_elements(0., zp)

    def set_focus(self, img):
        self.cm_ref = self.compute_com(img)
        zr = np.array(self.cm_ref).reshape(-1, 2)
        self.pid.set_point = self.model.predict(zr)

    def update(self, img):
        ncm = self.compute_com(img)
        if self.calculate_distance(ncm, self.cm_ref) > self.threshold:
            z = self.calculate_new_position(img)
            cz = self.pid.update(z)
            zp = self.ctd.data_list[-1] + cz
            self.ctd.add_elements(cz[0], zp[0])
            return True
        else:
            return False

    def calibrate(self, zs, img_stack, thd=0.04):
        nz, nx, ny = img_stack.shape
        cm = np.zeros((nz, 2))
        for i in range(nz):
            cm[i] = self.compute_com(img_stack[i])
        cm = cm.reshape(-1, 2)
        self.model.fit(cm, zs)
        diffs = np.diff(cm, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        self.threshold = thd * np.average(distances[:-1]) / np.abs(zs[0] - zs[1])

    def calculate_new_position(self, img):
        cm = self.compute_com(img)
        new_z = np.array(cm).reshape(-1, 2)
        return self.model.predict(new_z)

    def compute_com(self, img):
        img = self.background_filter(img)
        return ndimage.center_of_mass(img)

    @staticmethod
    def calculate_distance(center1, center2):
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance

    @staticmethod
    def crop_image(image, crop_size=(1024, 1024)):
        peak_coords = np.unravel_index(np.argmax(image, axis=None), image.shape)
        peak_row, peak_col = peak_coords
        crop_half_size = (crop_size[0] // 2, crop_size[1] // 2)
        start_row = max(peak_row - crop_half_size[0], 0)
        end_row = min(peak_row + crop_half_size[0], image.shape[0])
        start_col = max(peak_col - crop_half_size[1], 0)
        end_col = min(peak_col + crop_half_size[1], image.shape[1])
        # Adjust the crop size if it goes out of bounds
        if (end_row - start_row) < crop_size[0]:
            if start_row == 0:
                end_row = min(crop_size[0], image.shape[0])
            elif end_row == image.shape[0]:
                start_row = max(image.shape[0] - crop_size[0], 0)
        if (end_col - start_col) < crop_size[1]:
            if start_col == 0:
                end_col = min(crop_size[1], image.shape[1])
            elif end_col == image.shape[1]:
                start_col = max(image.shape[1] - crop_size[1], 0)
        # Crop the image
        cropped_image = image[start_row:end_row, start_col:end_col]
        return cropped_image

    @staticmethod
    def background_filter(img):
        thresh = threshold_otsu(img)
        binary = img > thresh
        return (img - thresh) * binary


class PID:
    """
    PID Controller
    Originally from IvPID. Author: Caner Durmusoglu
    """

    def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=50, Integrator_min=-50):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.Derivator = Derivator
        self.Integrator = Integrator
        self.Integrator_max = Integrator_max
        self.Integrator_min = Integrator_min
        self.set_point = 0.0
        self.error = 0.0

    def update(self, current_value):
        """
        Calculate PID output value for given reference input and feedback
        """
        self.error = self.set_point - current_value
        p_value = self.Kp * self.error
        d_value = self.Kd * (self.error - self.Derivator)
        self.Derivator = self.error

        self.Integrator += self.error
        self.Integrator = max(min(self.Integrator, self.Integrator_max), self.Integrator_min)
        i_value = self.Integrator * self.Ki

        return p_value + i_value + d_value

    @property
    def set_point(self):
        return self._set_point

    @set_point.setter
    def set_point(self, set_point):
        self._set_point = set_point
        self.Integrator = 0
        self.Derivator = 0
        self.error = 0.0

    @property
    def integrator(self):
        return self.Integrator

    @integrator.setter
    def integrator(self, Integrator):
        self.Integrator = Integrator

    @property
    def derivator(self):
        return self.Derivator

    @derivator.setter
    def derivator(self, Derivator):
        self.Derivator = Derivator

    @property
    def kp(self):
        return self.Kp

    @kp.setter
    def kp(self, P):
        self.Kp = P

    @property
    def ki(self):
        return self.Ki

    @ki.setter
    def ki(self, I):
        self.Ki = I

    @property
    def kd(self):
        return self.Kd

    @kd.setter
    def kd(self, D):
        self.Kd = D

    def get_error(self):
        return self.error


class CtrlData:

    def __init__(self, max_length):
        self.ctrl_list = deque(maxlen=max_length)
        self.data_list = deque(maxlen=max_length)

    def add_elements(self, ctrl, data):
        self.ctrl_list.extend([ctrl])
        self.data_list.extend([data])

    def get_elements(self):
        return np.array(self.ctrl_list.copy()) if self.ctrl_list else None, np.array(
            self.data_list.copy()) if self.data_list else None

    def get_last_elements(self):
        return self.ctrl_list[-1].copy() if self.ctrl_list else None, self.data_list[
            -1].copy() if self.data_list else None

    def reset(self):
        self.ctrl_list.clear()
        self.data_list.clear()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f = FocusLocker()
    f.initiate(50.2)
    f.set_focus(50.)
    for _ in range(100):
        control_value = f.pid.update(f.ctd.data_list[-1])
        f.ctd.add_elements(control_value, f.ctd.data_list[-1] + control_value)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f.ctd.data_list, label="Position")
    plt.axhline(y=f.pid.set_point, color='r', linestyle='--', label="Set Point")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Position Control using PID")
    plt.subplot(2, 1, 2)
    plt.plot(f.ctd.ctrl_list, label="Control Signal")
    plt.xlabel("Time")
    plt.ylabel("Control Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()
