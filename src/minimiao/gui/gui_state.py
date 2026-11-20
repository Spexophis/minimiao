# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Remember spin-box values between sessions.

Every QSpinBox / QDoubleSpinBox that a panel keeps as an attribute is saved on close as
"<panel>.<attribute>": value in a JSON file in the user's config folder, and restored when the
window opens. Restoring blocks the widgets' signals, so nothing is sent to a device; values are
read when used, exactly as if they had been typed. Widgets that no longer exist are ignored, and
Qt clamps restored values into each widget's current range.
"""

import json
import os
import sys

from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox

STATE_FILE = "gui_state.json"


def state_path():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "minimiao", STATE_FILE)


def _spin_boxes(panels: dict):
    for pname, panel in panels.items():
        for attr, w in vars(panel).items():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                yield f"{pname}.{attr}", w


def save(panels: dict, path=None):
    """Write all spin-box values; returns the file path."""
    path = path or state_path()
    values = {key: w.value() for key, w in _spin_boxes(panels)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"schema": "minimiao.gui_state/1", "spin_boxes": values}, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def restore(panels: dict, path=None) -> int:
    """Apply saved values (signals blocked); returns how many widgets were set. Missing file: 0."""
    path = path or state_path()
    try:
        with open(path) as f:
            values = json.load(f).get("spin_boxes", {})
    except (OSError, ValueError):
        return 0
    n = 0
    for key, w in _spin_boxes(panels):
        if key not in values:
            continue
        v = values[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        w.blockSignals(True)
        try:
            w.setValue(int(round(v)) if isinstance(w, QSpinBox) else float(v))
        finally:
            w.blockSignals(False)
        n += 1
    return n
