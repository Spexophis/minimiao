# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel


class TemperatureWaitDialog(QDialog):
    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Device Cooling Down")
        self.device = device

        layout = QVBoxLayout(self)
        self.label = QLabel("Waiting for device to reach 0°C...", self)
        self.temp_label = QLabel("", self)
        layout.addWidget(self.label)
        layout.addWidget(self.temp_label)

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(320, 100)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_temp)
        self.timer.start(1000)

        self.update_temp()

    def update_temp(self):
        temp = self.device.get_temperature()
        self.temp_label.setText(f"Current device temperature: {temp:.1f} °C")
        if temp >= 0:
            self.timer.stop()
            self.accept()
