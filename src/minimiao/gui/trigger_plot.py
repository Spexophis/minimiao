# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""Window with the time traces of a TriggerBox sequence (digital outputs + analog outputs)."""

import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from minimiao.devices import trigger_box as tbx

COLORS = ["#4FC3F7", "#FFB74D", "#81C784", "#E57373", "#BA68C8", "#FFF176", "#4DB6AC", "#F06292"]


class TriggerTraceWindow(QWidget):
    """Digital outputs stacked as 0/1 rows (top), analog outputs in volts (bottom); x linked, in ms."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trigger Box traces")
        self.resize(900, 520)
        layout = QVBoxLayout(self)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

    def show_sequence(self, seq, summary=None):
        dig, ana = tbx.traces(seq)
        self.glw.clear()
        desc = tbx.describe_outputs(summary) if summary else ""
        self.info.setText(f"SYSCLK {seq.sysclk_hz} Hz (tick {seq.tick_us:.1f} us), sequence {seq.duration_ms} ms"
                          + (f"\n{desc}" if desc else ""))
        p_dig = self.glw.addPlot(row=0, col=0)
        p_dig.setLabel("left", "digital outputs")
        p_dig.showGrid(x=True, y=False, alpha=0.3)
        p_dig.setDownsampling(auto=True, mode="peak")
        p_dig.setClipToView(True)
        outs = sorted(dig)
        ticks = []
        for row, out in enumerate(reversed(outs)):  # OUT with the lowest number on top
            t, y = dig[out]
            offset = 1.5 * row
            p_dig.plot(t, y + offset, pen=pg.mkPen(COLORS[(out - 1) % len(COLORS)], width=1.5))
            ticks.append((offset + 0.5, f"OUT{out}"))
        p_dig.getAxis("left").setTicks([ticks])
        if outs:
            p_dig.setYRange(-0.3, 1.5 * len(outs) - 0.2, padding=0)
        if ana:
            p_ana = self.glw.addPlot(row=1, col=0)
            p_ana.setLabel("left", "analog / V")
            p_ana.setLabel("bottom", "time / ms")
            p_ana.showGrid(x=True, y=True, alpha=0.3)
            p_ana.addLegend(offset=(10, 5))
            for ch in sorted(ana):
                t, v = ana[ch]
                p_ana.plot(t, v, pen=pg.mkPen(COLORS[(ch + 3) % len(COLORS)], width=1.5), name=f"AO#{ch}")
            p_ana.setYRange(-0.1, max(5.0, max(v.max() for _, v in ana.values())) + 0.1, padding=0)
            p_ana.setXLink(p_dig)
            self.glw.ci.layout.setRowStretchFactor(0, 2)
        else:
            p_dig.setLabel("bottom", "time / ms")
        p_dig.setXRange(0, max(seq.duration_ms, 1), padding=0.02)
        self.show()
        self.raise_()
        self.activateWindow()
