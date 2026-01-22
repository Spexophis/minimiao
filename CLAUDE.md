# CLAUDE.md - MiniMiao Codebase Guide for AI Assistants

> **Last Updated:** 2026-01-22
> **Project:** MiniMiao - Modular Microscope Control with Adaptive Optics
> **Version:** 1.0.0
> **License:** MIT

This document provides a comprehensive guide for AI assistants working with the MiniMiao codebase. It covers architecture, conventions, workflows, and critical context needed to make informed code changes.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Architecture & Key Components](#architecture--key-components)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Device Management](#device-management)
7. [GUI Framework](#gui-framework)
8. [Data Flow & Threading](#data-flow--threading)
9. [Configuration System](#configuration-system)
10. [Testing & Debugging](#testing--debugging)
11. [Common Tasks](#common-tasks)
12. [Critical Gotchas](#critical-gotchas)

---

## Project Overview

### What is MiniMiao?

MiniMiao is a professional-grade Python GUI application for controlling advanced microscopy systems. It combines:

- **Hardware Control:** 16+ device types (cameras, lasers, SLMs, piezo stages, DAQs, deformable mirrors)
- **Real-time Imaging:** Live acquisition with OpenGL-accelerated visualization
- **Advanced Imaging Modes:** Wide-field, SIM 2D/3D, point scanning, photon counting
- **Adaptive Optics:** Wavefront sensing and correction via deformable mirrors
- **Precision Synchronization:** 250 kHz DAQ timing for hardware coordination
- **Data Processing:** Real-time FFT, image reconstruction, focus analysis

### Technology Stack

| Category | Technologies |
|----------|-------------|
| **GUI** | PyQt6 6.9.1 (dark theme) |
| **Graphics** | PyOpenGL 3.1.10, PyQtGraph 0.14.0 |
| **Scientific** | NumPy 2.2.6, SciPy, scikit-image 0.25.2 |
| **Hardware** | NIDAQmx 1.3.0, ImagingControl4, pycobolt (git) |
| **Data I/O** | Tifffile 2025.5.10, Pandas 2.3.3, OpenPyXL 3.1.5 |
| **Build** | setuptools + wheel |

### Python Requirements

- **Minimum Version:** Python 3.10+
- **Package Manager:** uv (with locked dependencies in `uv.lock`)

---

## Codebase Structure

```
minimiao/
├── src/minimiao/              # Main package
│   ├── __init__.py           # Package initialization
│   ├── __main__.py           # CLI entry point (app() function)
│   ├── main.py               # AppWrapper - application orchestrator
│   ├── executor.py           # CommandExecutor - signal/hardware bridge (944 lines)
│   ├── run_threads.py        # Threading infrastructure
│   │
│   ├── devices/              # Hardware abstraction layer
│   │   ├── device.py         # DeviceManager + base classes
│   │   ├── andor_emccd.py    # Andor EMCCD camera (primary)
│   │   ├── mock_cam.py       # Mock camera for testing
│   │   ├── hamamatsu_scmos.py, flir_cmos.py, tis_cmos.py, thorlab_scmos.py, thorlab_webcam.py
│   │   ├── cobolt_laser.py   # Laser control
│   │   ├── fdd_slm.py, hamamatsu_slm.py  # Spatial light modulators
│   │   ├── mcl_deck.py       # XY stage
│   │   ├── mcl_piezo.py      # 3-axis piezo scanner (65KB - largest device)
│   │   ├── alpao_dm.py       # Deformable mirror (AO)
│   │   ├── ni_daq.py         # National Instruments DAQ
│   │   └── phaseform_dpp.py  # DPP device
│   │
│   ├── gui/                  # PyQt6 GUI components
│   │   ├── main_window.py    # QMainWindow with docked panels
│   │   ├── controller_panel.py  # Left dock (42KB) - primary controls
│   │   ├── viewer_window.py  # Central widget - live imaging
│   │   ├── ao_panel.py       # Right dock (22KB) - adaptive optics
│   │   ├── gl_viewer.py      # OpenGL image viewer (15KB)
│   │   ├── custom_widgets.py # Reusable custom widgets
│   │   └── cooldown_dialogue.py  # Camera cooling dialog
│   │
│   ├── computations/         # Signal processing & reconstruction
│   │   ├── trigger_generator.py      # TriggerSequence class
│   │   ├── image_reconstructions.py  # Point scan reconstruction
│   │   ├── shwfs_reconstruction.py   # Wavefront sensing
│   │   ├── focus_lock_control.py     # Focus locking algorithm
│   │   └── pattern_generator.py      # SLM phase patterns
│   │
│   └── utilities/            # Helper functions
│       ├── image_processor.py   # Focus measures, peak finding (13KB)
│       └── zernike_generator.py # Zernike polynomials (12KB)
│
├── pyproject.toml            # Project config + dependencies
├── uv.lock                   # Locked dependency versions (360KB)
├── README.md                 # Minimal project description
├── LICENSE                   # MIT License
└── .gitignore                # Standard Python ignores
```

**File Statistics:**
- **Total Python Files:** 36
- **Largest Module:** `executor.py` (944 lines, 42KB)
- **Total Package Size:** Well-organized, modular architecture

---

## Architecture & Key Components

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     PyQt6 Main Window                        │
│  ┌───────────────┬─────────────────────┬──────────────────┐  │
│  │  Controller   │    Live Viewer      │   AO Panel       │  │
│  │  Panel        │  (GL + PyQtGraph)   │  (Zernike)       │  │
│  │  - Camera ROI │  - Image display    │  - Wavefront     │  │
│  │  - Laser pwr  │  - FFT analysis     │  - DM control    │  │
│  │  - Piezo pos  │  - Photon counts    │  - Aberrations   │  │
│  │  - Acq modes  │  - Profile plots    │                  │  │
│  └───────┬───────┴──────────┬──────────┴────────┬─────────┘  │
└──────────┼──────────────────┼───────────────────┼────────────┘
           │ Signals          │ Display           │ Controls
      ┌────▼──────────────────▼───────────────────▼──────────┐
      │           CommandExecutor (executor.py)              │
      │  - Orchestrates all hardware operations              │
      │  - Connects PyQt6 signals to device methods          │
      │  - Manages acquisition modes and data saving         │
      └────┬──────────┬─────────────────┬─────────────────────┘
           │          │                 │
    ┌──────▼─────┐ ┌─▼────────────┐ ┌──▼────────────┐
    │  Device    │ │ Computation  │ │  Threading    │
    │  Manager   │ │  Manager     │ │  Workers      │
    │            │ │              │ │               │
    │ - 16 types │ │ - TriggerSeq │ │ - CameraAcq   │
    │ - Init     │ │ - ImgRecon   │ │ - PhotonCount │
    │ - Fallback │ │ - SHWFS      │ │ - TaskWorker  │
    │            │ │ - FocusLock  │ │ - FFTWorker   │
    └──────┬─────┘ └─┬────────────┘ └──┬────────────┘
           │         │                 │
    ┌──────▼─────────▼─────────────────▼──────────┐
    │        Hardware & Data Layer                │
    │                                             │
    │  Cameras • Lasers • SLM • Piezo • DAQ •    │
    │  Deformable Mirror • TIFF • Excel • Logger │
    └─────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. **AppWrapper** (`main.py`)
**Purpose:** Application lifecycle manager

**Responsibilities:**
- Initialize PyQt6 QApplication with dark theme
- Setup folder structure: `~/Documents/data/YYYYMMDD_username/`
- Configure JSON logging to timestamped files
- Load user configuration from JSON file
- Create three managers: DeviceManager, ComputationManager, MainWindow
- Instantiate CommandExecutor to wire signals
- Start Qt event loop

**Entry Point:** `__main__.py::app()` → `AppWrapper.run()`

#### 2. **CommandExecutor** (`executor.py`)
**Purpose:** Central controller bridging GUI and hardware

**Key Methods:**
- **Camera:** `camera_temperature_check()`, `cooler_on()`, `cooler_off()`, `set_roi()`, `set_gain()`
- **Piezo:** `piezo_usb_move()`, `piezo_analog_move()`, `focus_find()`
- **Laser:** `laser_on()`, `laser_off()`, `set_laser_power()`
- **Imaging:** `video()`, `start_video()`, `stop_video()`, `fft_analysis()`
- **Acquisition:** `prepare_point_scan()`, `prepare_wide_field()`, `prepare_sim_2d()`, `start_triggers()`
- **Data:** `save_tiff()`, `save_metadata()`, `save_photon_counts()`

**Critical:** This is the **main orchestrator** - all user actions flow through here.

#### 3. **DeviceManager** (`devices/device.py`)
**Purpose:** Hardware abstraction and initialization

**Device Categories:**
- **Cameras (7 types):** EMCCD (primary), sCMOS, CMOS, webcam, mock
- **Light Sources:** Cobolt lasers, SLMs (FDD, Hamamatsu)
- **Positioning:** MCL Deck (XY stage), MCL Piezo (3-axis scanner)
- **Adaptive Optics:** ALPAO deformable mirror
- **DAQ:** National Instruments (triggers, photon counting)

**Initialization Flow:**
1. Parse configuration JSON
2. Attempt to connect to each device
3. On failure, log error and continue (graceful degradation)
4. Special case: EMCCD failure → automatic fallback to MockCamera

**Key Pattern:** All devices inherit from base classes with standard interfaces

#### 4. **ComputationManager** (`computations/`)
**Purpose:** Real-time processing and reconstruction

**Components:**
- **TriggerSequence:** Generate DAQ waveforms for synchronized hardware
- **ImageReconstructions:** Point scan → 2D image conversion
- **SHWFS:** Shack-Hartmann wavefront sensor processing
- **FocusLock:** Z-axis feedback control
- **PatternGenerator:** SLM phase masks

#### 5. **Threading Infrastructure** (`run_threads.py`)
**Purpose:** Non-blocking concurrent operations

**Thread Classes:**
| Thread | Purpose | Data Structure |
|--------|---------|----------------|
| `CameraAcquisitionThread` | Continuous camera polling | `CameraDataList` (circular buffer) |
| `PhotonCountThread` | Photon counter acquisition | `PhotonCountList` (aggregator) |
| `TaskWorker` (QThread) | Long-running tasks | Progress dialog integration |
| `FFTWorker` | Real-time FFT computation | Emits signals to viewer |
| `PSLiveWorker` | Point scan reconstruction | Live image updates |

**Thread Safety:** Double-buffering pattern for concurrent read/write

---

## Development Workflows

### Application Startup Sequence

```
User runs: python -m minimiao

1. __main__.py::app()
2. main.py::AppWrapper()
3. AppWrapper.run()
   ├─ Create QApplication (dark theme)
   ├─ Setup logging: ~/Documents/data/YYYYMMDD_username/YYYYMMDD_HHMM.log
   ├─ File dialog: Select configuration JSON
   ├─ DeviceManager.__init__(config)
   │   ├─ Connect to EMCCD (or fallback to MockCamera)
   │   ├─ Connect to Cobolt laser
   │   ├─ Connect to MCL Piezo
   │   ├─ Connect to NI DAQ
   │   └─ ... (all configured devices)
   ├─ ComputationManager.__init__(config)
   │   ├─ TriggerSequence
   │   ├─ ImageReconstructions
   │   └─ SHWFS
   ├─ MainWindow.__init__(config)
   │   ├─ ControllerPanel (left dock)
   │   ├─ ViewerWindow (central)
   │   └─ AOPanel (right dock)
   ├─ CommandExecutor(managers)
   │   └─ Connect all signals/slots
   └─ mainWindow.show() + app.exec()
```

### Common Acquisition Workflows

#### Wide-Field Imaging
```
1. User configures camera ROI, gain, laser power
2. Click "Wide Field" mode
3. CommandExecutor.prepare_wide_field()
   ├─ Configure camera (ROI, exposure)
   ├─ Set laser power
   ├─ Generate digital trigger sequence
   └─ Setup DAQ timing
4. Click "Start"
5. CommandExecutor.start_triggers()
   ├─ Write triggers to DAQ
   ├─ Start camera acquisition
   └─ Capture single frame
6. Save TIFF + metadata Excel
```

#### Point Scan 2D (Live Reconstruction)
```
1. Configure point scan parameters (step size, range)
2. Click "Point Scan 2D" mode
3. CommandExecutor.prepare_point_scan()
   ├─ Generate piezo raster scan sequence
   ├─ Generate photon counter gate triggers
   ├─ Configure DAQ sample rate (250 kHz)
   └─ Initialize ImageReconstructions
4. Click "Start"
5. CommandExecutor.start_triggers() + photon_counter()
   ├─ DAQ outputs piezo waveform
   ├─ Piezo scans XY plane
   ├─ Photon counter acquires counts
   └─ PSLiveWorker reconstructs in real-time
6. Live display updates (photon count image + traces)
7. Save: TIFF (image) + NPY (photon counts) + XLSX (metadata)
```

#### Focus Finding Algorithm
```
1. Click "Focus Finding"
2. CommandExecutor.prepare_focus_finding()
   ├─ Define Z-stack range (e.g., -5 to +5 µm, 0.5 µm steps)
   ├─ Configure camera for fast acquisition
   └─ Prepare focus measure function (Sobel filter)
3. CommandExecutor.focus_find()
   ├─ For each Z position:
   │   ├─ Move piezo to Z
   │   ├─ Acquire frame
   │   ├─ Calculate focus measure
   │   └─ Store (Z, focus_value, image)
   ├─ Find Z_max (peak focus measure)
   └─ Move piezo to Z_max
4. Save Z-stack as multi-page TIFF
5. Plot focus measure vs. Z position
```

### Data Saving Conventions

**File Organization:**
```
~/Documents/data/YYYYMMDD_username/
├── YYYYMMDD_HHMM.log                    # Application log
├── YYYYMMDD_HHMM_wide_field_001.tif    # Image data
├── YYYYMMDD_HHMM_wide_field_001.xlsx   # Metadata
├── YYYYMMDD_HHMM_point_scan_002.tif    # Reconstructed image
├── YYYYMMDD_HHMM_point_scan_002.npy    # Raw photon counts
└── YYYYMMDD_HHMM_point_scan_002.xlsx   # Acquisition parameters
```

**Metadata (Excel) Includes:**
- Timestamp
- Acquisition mode (wide field, SIM, point scan)
- Camera settings (ROI, gain, exposure)
- Laser power
- Scan parameters (range, step size, speed)
- Trigger sequence details

---

## Code Conventions

### Naming Conventions

| Type | Convention | Examples |
|------|-----------|----------|
| **Classes** | PascalCase | `DeviceManager`, `CommandExecutor`, `TriggerSequence` |
| **Functions** | snake_case | `prepare_video()`, `start_triggers()`, `focus_find()` |
| **Methods** | snake_case | `camera_temperature_check()`, `piezo_usb_move()` |
| **Constants** | UPPER_SNAKE_CASE | `SAMPLE_RATE`, `MAX_LASER_POWER` |
| **Private** | _leading_underscore | `_internal_method()`, `_buffer` |
| **Devices** | snake_case attributes | `self.emccd`, `self.cobolt_laser`, `self.mcl_piezo` |

### Import Organization

**Standard Order:**
1. Python standard library
2. Third-party packages (numpy, PyQt6, etc.)
3. Local modules (relative imports)

**Example:**
```python
import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMainWindow

from .devices.device import DeviceManager
from .computations.trigger_generator import TriggerSequence
```

### PyQt6 Signal/Slot Patterns

**Signal Naming:**
- Use descriptive past-tense: `laser_power_changed`, `acquisition_finished`, `roi_updated`

**Signal Definition:**
```python
class ControllerPanel(QWidget):
    laser_power_changed = pyqtSignal(float)  # power in mW
    roi_updated = pyqtSignal(int, int, int, int)  # x, y, width, height
```

**Connection in CommandExecutor:**
```python
def __init__(self, device_manager, computation_manager, main_window):
    # Connect signals to slots
    controller = main_window.controller_panel
    controller.laser_power_changed.connect(self.set_laser_power)
    controller.roi_updated.connect(self.set_roi)
```

### Error Handling

**Device Initialization:**
```python
try:
    self.emccd = AndorEMCCD(config)
    logging.info("EMCCD initialized successfully")
except Exception as e:
    logging.error(f"EMCCD initialization failed: {e}")
    self.emccd = MockCamera(config)
    logging.info("Fallback to MockCamera")
```

**Critical Principle:** **Never crash the application** due to hardware failure. Always log and gracefully degrade.

### Logging Standards

**Configuration:**
- JSON format
- Timestamped files: `YYYYMMDD_HHMM.log`
- Location: `~/Documents/data/YYYYMMDD_username/`

**Log Levels:**
```python
logging.debug("Detailed diagnostic info")
logging.info("Routine operations (device init, acquisition start)")
logging.warning("Unexpected but handled (device timeout, retry)")
logging.error("Serious issue (device failure, data save error)")
logging.critical("Application-breaking error (rare)")
```

**Best Practice:** Log all device operations, configuration changes, and user actions for debugging.

---

## Device Management

### Device Lifecycle

**Initialization (DeviceManager):**
1. Parse configuration JSON
2. Instantiate device class with config
3. Call `connect()` or `initialize()` method
4. Store reference in DeviceManager attributes
5. Log success/failure

**Configuration (JSON):**
```json
{
  "camera": {
    "type": "andor_emccd",
    "roi": [512, 512, 1024, 1024],
    "binning": 1,
    "gain": 100,
    "target_temperature": -70
  },
  "laser": {
    "type": "cobolt",
    "wavelengths": [488, 561, 640],
    "max_power": 100
  },
  "piezo": {
    "type": "mcl_piezo",
    "step_sizes": [0.032, 0.032, 0.16],
    "ranges": [2.0, 2.0, 0.0]
  }
}
```

### Camera Abstraction

**Base Interface (all cameras must implement):**
```python
class BaseCamera:
    def set_roi(self, x, y, width, height):
        """Set region of interest"""

    def set_exposure(self, exposure_ms):
        """Set exposure time in milliseconds"""

    def start_acquisition(self):
        """Begin continuous acquisition"""

    def get_latest_frame(self):
        """Return most recent frame as numpy array"""

    def stop_acquisition(self):
        """Stop acquisition and cleanup"""
```

**Implemented Cameras:**
1. **AndorEMCCD** (`andor_emccd.py`) - Primary camera
   - Cooler control (target: -70°C)
   - 14-bit depth, 2048x2048 sensor
   - EMCCD gain adjustment

2. **MockCamera** (`mock_cam.py`) - Testing fallback
   - Generates random 2048x2048 images
   - 14-bit simulated data
   - No hardware required

**Usage Pattern:**
```python
# In CommandExecutor
camera = self.device_manager.emccd
camera.set_roi(512, 512, 1024, 1024)
camera.set_exposure(50)  # 50 ms
camera.start_acquisition()

# In CameraAcquisitionThread
while self.running:
    frame = camera.get_latest_frame()
    self.data_list.add_frame(frame)
```

### DAQ (National Instruments)

**Purpose:** Precision hardware synchronization

**Capabilities:**
- **Digital Output:** 6 channels for trigger signals
- **Analog Output:** Piezo scanner control (X, Y, Z)
- **Counter Input:** Photon counting
- **Sample Rate:** 250 kHz (configurable)

**Trigger Generation:**
```python
trigger_seq = TriggerSequence(config)
trigger_seq.set_mode("point_scan_2d")
trigger_seq.generate()

digital_triggers = trigger_seq.get_digital_channels()  # 6 channels
analog_piezo = trigger_seq.get_analog_channels()      # 3 channels (X, Y, Z)

daq = self.device_manager.ni_daq
daq.write_digital_triggers(digital_triggers)
daq.write_analog_waveforms(analog_piezo)
daq.start()
```

**Channel Assignments (typical):**
- Digital 0: Camera trigger
- Digital 1: Laser gate
- Digital 2: SLM sync
- Digital 3: Photon counter gate
- Digital 4-5: Reserved
- Analog 0-2: Piezo X, Y, Z

### Piezo Scanner (MCL NanoDrive)

**Control Modes:**
1. **USB Mode:** Direct positioning (slow, ~10 Hz)
2. **Analog Mode:** DAQ-driven scanning (fast, synchronized)

**Usage:**
```python
piezo = self.device_manager.mcl_piezo

# USB mode: Direct positioning
piezo.move_usb(x=1.5, y=2.0, z=0.5)  # Absolute position in mm

# Analog mode: Scan sequence
scan_waveform = trigger_seq.get_piezo_scan()
daq.write_analog_waveforms(scan_waveform)
daq.start()  # Piezo follows DAQ waveform
```

**Raster Scan Pattern:**
```
Y  ┌─→─→─→─→─→┐
   │          ↓
   ↑ ←─←─←─←─←┘
   │  ┌─→─→─→─→┐
   └──┘        ↓
              X
```

**Parameters:**
- Step size: [32 nm, 32 nm, 160 nm] (X, Y, Z)
- Range: [2.0 mm, 2.0 mm, 0.0 mm]
- Bidirectional scanning supported

---

## GUI Framework

### Window Structure

**MainWindow** (QMainWindow):
- **Left Dock:** ControllerPanel (42KB) - primary user interface
- **Central Widget:** ViewerWindow - live imaging display
- **Right Dock:** AOPanel (22KB) - adaptive optics controls

**Dark Theme:** Applied at QApplication level in `AppWrapper`

### ControllerPanel (`controller_panel.py`)

**Major Sections:**
1. **Camera Group:**
   - ROI spinboxes (x, y, width, height)
   - Binning selector
   - Gain slider
   - Temperature display + cooler controls

2. **Laser Group:**
   - Wavelength selector (488/561/640 nm)
   - Power spinbox (0-100 mW)
   - On/Off button

3. **Piezo Group:**
   - X/Y/Z position spinboxes
   - USB Move button
   - Focus Finding button

4. **Acquisition Modes:**
   - Radio buttons: Wide Field / SIM 2D / SIM 3D / Point Scan 2D
   - Start/Stop buttons
   - Video On/Off toggle

5. **Data Saving:**
   - Filename prefix
   - Save button
   - Metadata options

**State Persistence:**
- Spinbox values saved to `~/.config/minimiao/last_session.json`
- Auto-restored on startup

### ViewerWindow (`viewer_window.py`)

**Display Modes (tabs):**
1. **Live Image** - GLViewer (OpenGL) for fast rendering
2. **FFT** - Real-time Fourier transform
3. **Profiles** - Line profiles and traces
4. **Photon Counts** - Point scan visualization

**Colormap Options:**
- Grayscale, Viridis, Plasma, Inferno, Magma
- Auto-scaling or manual min/max

**Overlays:**
- ROI rectangle
- Crosshair
- Scale bar

### Custom Widgets (`custom_widgets.py`)

**Reusable Components:**
- `DoubleSpinBoxWithLabel` - Labeled numeric input
- `SliderWithValue` - Slider + value display
- `LogViewerWidget` - Real-time log display
- `ProgressBar` - Custom styled progress indicator

**Usage Example:**
```python
from .custom_widgets import DoubleSpinBoxWithLabel

laser_power = DoubleSpinBoxWithLabel(
    label="Laser Power (mW)",
    minimum=0.0,
    maximum=100.0,
    decimals=2,
    step=0.1
)
laser_power.valueChanged.connect(self.on_power_changed)
```

---

## Data Flow & Threading

### Thread Architecture

```
Main Thread (Qt Event Loop)
├─ GUI event handling
├─ Signal/slot connections
└─ CommandExecutor method calls
    │
    ├─→ CameraAcquisitionThread (background)
    │   ├─ Polls camera.get_latest_frame()
    │   ├─ Adds to CameraDataList (thread-safe)
    │   └─ Emits frame_ready signal
    │
    ├─→ PhotonCountThread (background)
    │   ├─ Polls photon_counter.read()
    │   ├─ Aggregates to PhotonCountList
    │   └─ Emits data_ready signal
    │
    ├─→ TaskWorker (QThread for long tasks)
    │   ├─ Runs acquisition sequences
    │   ├─ Shows progress dialog
    │   └─ Emits finished signal
    │
    ├─→ FFTWorker (real-time processing)
    │   ├─ Computes np.fft.fft2(frame)
    │   ├─ Applies fftshift and log scaling
    │   └─ Emits fft_computed signal → ViewerWindow
    │
    └─→ PSLiveWorker (point scan reconstruction)
        ├─ Gets photon counts from PhotonCountList
        ├─ Calls ImageReconstructions.reconstruct()
        ├─ Generates 2D image from scan data
        └─ Emits image_reconstructed signal → ViewerWindow
```

### Thread-Safe Data Structures

**CameraDataList:**
```python
class CameraDataList:
    def __init__(self, max_size=100):
        self._data = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add_frame(self, frame):
        with self._lock:
            self._data.append(frame)
            if len(self._data) > self._max_size:
                self._data.pop(0)  # Circular buffer

    def get_latest(self):
        with self._lock:
            return self._data[-1] if self._data else None
```

**Critical Pattern:** Always use locks for shared data between threads.

### Signal Flow Example: Live Video

```
1. User clicks "Video ON" button (ControllerPanel)
2. Signal: controller_panel.video_toggled(True)
3. CommandExecutor.video(True)
   ├─ prepare_video()
   │   ├─ Set camera ROI, exposure
   │   ├─ Set laser power
   │   └─ Generate DAQ triggers
   ├─ start_video()
   │   ├─ camera.start_acquisition()
   │   ├─ CameraAcquisitionThread.start()
   │   └─ daq.start()
   └─ camera_thread.frame_ready.connect(viewer.display_frame)
4. CameraAcquisitionThread runs:
   while running:
       frame = camera.get_latest_frame()
       camera_data_list.add_frame(frame)
       emit frame_ready(frame)
5. ViewerWindow.display_frame(frame)
   ├─ gl_viewer.set_image(frame)
   └─ Render via OpenGL
6. User clicks "Video OFF"
7. stop_video()
   ├─ camera_thread.stop()
   └─ camera.stop_acquisition()
```

---

## Configuration System

### Configuration Files

**Location:** User-selected at startup via file dialog

**Format:** JSON with nested structure

**Complete Example:**
```json
{
  "metadata": {
    "created": "2026-01-22",
    "description": "Standard EMCCD + AO configuration"
  },

  "camera": {
    "type": "andor_emccd",
    "roi": [512, 512, 1024, 1024],
    "binning": 1,
    "gain": 100,
    "exposure_ms": 50,
    "target_temperature": -70,
    "cooling_timeout_s": 600
  },

  "laser": {
    "type": "cobolt",
    "port": "COM3",
    "wavelengths": [488, 561, 640],
    "max_power": 100,
    "default_power": 10
  },

  "slm": {
    "type": "fdd_slm",
    "resolution": [2048, 1536],
    "phase_patterns": {
      "order_1": "patterns/slm_order1.npy",
      "order_2": "patterns/slm_order2.npy"
    }
  },

  "piezo": {
    "type": "mcl_piezo",
    "usb_device_id": 0,
    "step_sizes": [0.032, 0.032, 0.16],
    "ranges": [2.0, 2.0, 0.0],
    "conversion_factors": [10.0, 10.0, 50.0]
  },

  "daq": {
    "type": "ni_daq",
    "device_name": "Dev1",
    "sample_rate": 250000,
    "digital_channels": ["Dev1/port0/line0:5"],
    "analog_channels": ["Dev1/ao0:2"],
    "counter_channel": "Dev1/ctr0"
  },

  "deformable_mirror": {
    "type": "alpao_dm",
    "serial_number": "BAX123",
    "num_actuators": 97,
    "calibration_file": "calibration/alpao_BAX123.fits"
  },

  "acquisition": {
    "data_directory": "~/Documents/data",
    "default_mode": "wide_field",
    "autosave": true
  },

  "zernike_modes": {
    "max_order": 10,
    "modes_to_correct": [4, 5, 6, 7, 8, 9, 10, 11]
  }
}
```

### Configuration Loading

**In AppWrapper:**
```python
def run(self):
    # File dialog for config selection
    config_path, _ = QFileDialog.getOpenFileName(
        None, "Select Configuration", "", "JSON Files (*.json)"
    )

    with open(config_path) as f:
        config = json.load(f)

    # Pass to all managers
    self.device_manager = DeviceManager(config)
    self.computation_manager = ComputationManager(config)
    self.main_window = MainWindow(config)
```

**Accessing Configuration:**
```python
# In device initialization
class AndorEMCCD:
    def __init__(self, config):
        camera_config = config['camera']
        self.target_temp = camera_config['target_temperature']
        self.default_roi = camera_config['roi']
```

### Session State Persistence

**Saved on Exit:**
- Spinbox values (ROI, laser power, piezo positions)
- Window positions and dock states
- Last used configuration path

**Location:** `~/.config/minimiao/last_session.json`

**Auto-Restore on Startup:** Yes (if file exists)

---

## Testing & Debugging

### Current Testing Setup

**Status:** No formal test suite (pytest, unittest)

**Testing Approach:**
1. **Mock Hardware:** `mock_cam.py` for GUI/workflow testing
2. **Manual Testing:** Run application and interact with GUI
3. **Logging:** Comprehensive JSON logs for debugging

### MockCamera Capabilities

**Purpose:** Enable testing without physical hardware

**Features:**
- Generates random 2048x2048 images (14-bit depth)
- Simulates exposure time delays
- Configurable frame rate
- Same interface as real cameras

**Usage:**
```python
# Automatic fallback
try:
    camera = AndorEMCCD(config)
except:
    camera = MockCamera(config)  # Graceful degradation
```

### Debugging Techniques

#### Enable Verbose Logging

**In AppWrapper:**
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()  # Also print to console
    ]
)
```

#### Common Issues & Solutions

**Issue: Camera Not Detected**
- **Check:** Device powered on, USB connected
- **Check:** Configuration JSON has correct device type
- **Check:** Logs for initialization errors
- **Solution:** Application will fallback to MockCamera

**Issue: DAQ Timing Errors**
- **Check:** Sample rate vs. waveform length
- **Check:** Digital/analog channel assignments
- **Cause:** Often mismatch between trigger duration and acquisition time
- **Solution:** Verify `TriggerSequence` parameters match scan duration

**Issue: GUI Freezing During Acquisition**
- **Cause:** Long operation in main thread
- **Solution:** Move to `TaskWorker` with progress dialog

**Issue: Point Scan Reconstruction Slow**
- **Check:** `PSLiveWorker` thread priority
- **Check:** Image size (reduce for faster processing)
- **Optimization:** Use `numba` JIT compilation for hot loops

#### Performance Profiling

**For Camera Acquisition:**
```python
import time

start = time.perf_counter()
frame = camera.get_latest_frame()
elapsed = time.perf_counter() - start
logging.debug(f"Frame acquisition: {elapsed*1000:.2f} ms")
```

**For FFT Computation:**
```python
import cProfile

profiler = cProfile.Profile()
profiler.enable()
fft_result = np.fft.fft2(image)
profiler.disable()
profiler.print_stats(sort='cumtime')
```

---

## Common Tasks

### Adding a New Device

**Step 1:** Create device file in `devices/`
```python
# devices/new_device.py
import logging

class NewDevice:
    def __init__(self, config):
        self.config = config['new_device']
        self.connected = False

    def connect(self):
        try:
            # Device-specific initialization
            self.connected = True
            logging.info("NewDevice connected successfully")
        except Exception as e:
            logging.error(f"NewDevice connection failed: {e}")
            raise

    def disconnect(self):
        # Cleanup
        self.connected = False
```

**Step 2:** Add to DeviceManager
```python
# devices/device.py
from .new_device import NewDevice

class DeviceManager:
    def __init__(self, config):
        # Existing devices...

        # Add new device
        try:
            self.new_device = NewDevice(config)
            self.new_device.connect()
        except Exception as e:
            logging.error(f"NewDevice initialization failed: {e}")
            self.new_device = None
```

**Step 3:** Add configuration schema
```json
{
  "new_device": {
    "type": "new_device",
    "parameter1": "value1",
    "parameter2": 42
  }
}
```

**Step 4:** Add control methods to CommandExecutor
```python
# executor.py
def new_device_action(self, param):
    if self.device_manager.new_device:
        self.device_manager.new_device.do_action(param)
    else:
        logging.warning("NewDevice not available")
```

### Adding a New Imaging Mode

**Step 1:** Define mode in TriggerSequence
```python
# computations/trigger_generator.py
class TriggerSequence:
    def generate(self):
        if self.mode == "new_mode":
            self._generate_new_mode()

    def _generate_new_mode(self):
        # Create digital trigger waveforms
        self.digital_triggers = np.zeros((6, num_samples))
        # ... configure triggers
```

**Step 2:** Add preparation method to CommandExecutor
```python
# executor.py
def prepare_new_mode(self):
    logging.info("Preparing new acquisition mode")

    # Configure camera
    self.set_roi(512, 512, 1024, 1024)
    self.set_exposure(50)

    # Configure laser
    self.set_laser_power(10)

    # Generate triggers
    trigger_seq = self.computation_manager.trigger_sequence
    trigger_seq.set_mode("new_mode")
    trigger_seq.generate()

    # Configure DAQ
    daq = self.device_manager.ni_daq
    daq.set_sample_rate(250000)
    daq.write_digital_triggers(trigger_seq.get_digital_channels())
```

**Step 3:** Add GUI controls in ControllerPanel
```python
# gui/controller_panel.py
new_mode_radio = QRadioButton("New Mode")
new_mode_radio.toggled.connect(
    lambda: self.mode_changed.emit("new_mode")
)
```

**Step 4:** Connect signal in CommandExecutor
```python
# executor.py
controller.mode_changed.connect(self.on_mode_changed)

def on_mode_changed(self, mode):
    if mode == "new_mode":
        self.prepare_new_mode()
```

### Modifying GUI Layout

**Example: Adding a New Control to ControllerPanel**

```python
# gui/controller_panel.py
class ControllerPanel(QWidget):
    # Define signal
    new_parameter_changed = pyqtSignal(float)

    def __init__(self, config):
        super().__init__()

        # Create widget
        self.new_param_spinbox = DoubleSpinBoxWithLabel(
            label="New Parameter",
            minimum=0.0,
            maximum=100.0,
            decimals=2,
            step=0.1,
            initial_value=10.0
        )

        # Connect signal
        self.new_param_spinbox.valueChanged.connect(
            self.new_parameter_changed.emit
        )

        # Add to layout
        control_group = QGroupBox("New Controls")
        layout = QVBoxLayout()
        layout.addWidget(self.new_param_spinbox)
        control_group.setLayout(layout)

        # Add to main layout
        self.main_layout.addWidget(control_group)
```

**Connect in CommandExecutor:**
```python
controller.new_parameter_changed.connect(self.on_new_parameter_changed)

def on_new_parameter_changed(self, value):
    logging.info(f"New parameter changed to {value}")
    # Do something with the value
```

---

## Critical Gotchas

### 1. **Thread Safety in GUI Updates**

**Problem:** Cannot update GUI widgets from background threads

**Symptom:** `QObject::setProperty: Cannot set property from different thread`

**Solution:** Use signals to communicate with main thread
```python
# WRONG - crashes
class MyThread(QThread):
    def run(self):
        label.setText("Updated")  # ERROR!

# CORRECT - use signals
class MyThread(QThread):
    update_signal = pyqtSignal(str)

    def run(self):
        self.update_signal.emit("Updated")

# In main thread
thread.update_signal.connect(label.setText)
```

### 2. **DAQ Timing Mismatch**

**Problem:** Trigger sequence length ≠ acquisition duration

**Symptom:** Incomplete scans, missing frames, DAQ buffer errors

**Example:**
```python
# WRONG
scan_duration = 1.0  # seconds
sample_rate = 250000  # Hz
num_samples = 100000  # Only 0.4 seconds!

# CORRECT
num_samples = int(scan_duration * sample_rate)  # 250000 samples
```

**Verification:**
```python
trigger_seq.generate()
assert len(trigger_seq.digital_triggers[0]) == expected_samples
```

### 3. **Camera ROI Out of Bounds**

**Problem:** Setting ROI larger than sensor

**Symptom:** Camera driver errors, crashes

**Solution:** Always validate ROI
```python
def set_roi(self, x, y, width, height):
    max_width, max_height = self.sensor_size

    # Validate
    if x + width > max_width or y + height > max_height:
        logging.error(f"ROI out of bounds: {x},{y},{width},{height}")
        return False

    # Apply
    self.camera.set_roi(x, y, width, height)
    return True
```

### 4. **Circular Import Errors**

**Problem:** Modules importing each other

**Symptom:** `ImportError: cannot import name 'X' from partially initialized module`

**Common Cause:**
```python
# devices/device.py
from ..executor import CommandExecutor  # Circular!

# executor.py
from .devices.device import DeviceManager  # Circular!
```

**Solution:** Use dependency injection
```python
# executor.py
class CommandExecutor:
    def __init__(self, device_manager):
        self.device_manager = device_manager  # Pass as parameter

# main.py
device_manager = DeviceManager(config)
executor = CommandExecutor(device_manager)  # Inject dependency
```

### 5. **File Path Issues (Cross-Platform)**

**Problem:** Hardcoded paths with `\` on Windows fail on Linux

**Solution:** Use `pathlib.Path`
```python
# WRONG
data_dir = "C:\\Users\\data\\images"  # Windows-only

# CORRECT
from pathlib import Path
data_dir = Path.home() / "Documents" / "data" / "images"
data_dir.mkdir(parents=True, exist_ok=True)
```

### 6. **NumPy Array Byte Order**

**Problem:** Some devices expect specific byte order for image data

**Symptom:** Corrupted images, incorrect pixel values

**Solution:** Explicitly set dtype
```python
# Camera returns big-endian uint16
frame = camera.get_latest_frame()  # dtype='>u2'

# Convert to native byte order for processing
frame_native = frame.astype(np.uint16)  # dtype='<u2' on little-endian
```

### 7. **PyQt6 QThread vs Python threading.Thread**

**Problem:** Mixing Qt and Python threads causes signal issues

**Rule:** Use `QThread` for threads that emit Qt signals

```python
# CORRECT - emits Qt signals
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

# WRONG - Python thread cannot emit Qt signals reliably
class CameraThread(threading.Thread):
    frame_ready = pyqtSignal(np.ndarray)  # Won't work!
```

### 8. **Resource Cleanup on Exit**

**Problem:** Devices not disconnected properly on application exit

**Symptom:** Devices stay in use, cannot reconnect

**Solution:** Implement cleanup in AppWrapper
```python
# main.py
class AppWrapper:
    def run(self):
        try:
            self.app.exec()
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("Cleaning up devices...")

        # Stop acquisition threads
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
            self.camera_thread.wait()

        # Disconnect devices
        if hasattr(self.device_manager, 'emccd'):
            self.device_manager.emccd.disconnect()

        logging.info("Cleanup complete")
```

### 9. **Configuration Validation**

**Problem:** Missing or invalid configuration keys crash application

**Solution:** Validate configuration on load
```python
REQUIRED_KEYS = ['camera', 'laser', 'daq', 'acquisition']

def validate_config(config):
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Type validation
    if not isinstance(config['camera']['roi'], list):
        raise ValueError("camera.roi must be a list")

    if len(config['camera']['roi']) != 4:
        raise ValueError("camera.roi must have 4 elements [x, y, w, h]")
```

### 10. **Memory Leaks in Live Acquisition**

**Problem:** Circular buffer grows unbounded

**Symptom:** Memory usage increases over time, eventually crashes

**Solution:** Implement proper circular buffer
```python
class CameraDataList:
    def __init__(self, max_size=100):
        self._max_size = max_size

    def add_frame(self, frame):
        with self._lock:
            self._data.append(frame)

            # CRITICAL: Remove old frames
            while len(self._data) > self._max_size:
                self._data.pop(0)
```

---

## Quick Reference

### File Locations

| Purpose | Path |
|---------|------|
| **Entry Point** | `src/minimiao/__main__.py::app()` |
| **Main Orchestrator** | `src/minimiao/executor.py` |
| **Device Manager** | `src/minimiao/devices/device.py` |
| **Camera Primary** | `src/minimiao/devices/andor_emccd.py` |
| **Mock Camera** | `src/minimiao/devices/mock_cam.py` |
| **Main Window** | `src/minimiao/gui/main_window.py` |
| **Controller Panel** | `src/minimiao/gui/controller_panel.py` (42KB) |
| **Viewer Window** | `src/minimiao/gui/viewer_window.py` |
| **Trigger Generator** | `src/minimiao/computations/trigger_generator.py` |
| **Threading** | `src/minimiao/run_threads.py` |

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `AppWrapper` | `main.py` | Application lifecycle |
| `CommandExecutor` | `executor.py` | Signal/hardware bridge |
| `DeviceManager` | `devices/device.py` | Hardware abstraction |
| `TriggerSequence` | `computations/trigger_generator.py` | DAQ waveforms |
| `ImageReconstructions` | `computations/image_reconstructions.py` | Point scan → image |
| `MainWindow` | `gui/main_window.py` | Qt main window |
| `ControllerPanel` | `gui/controller_panel.py` | User controls |
| `ViewerWindow` | `gui/viewer_window.py` | Image display |
| `CameraAcquisitionThread` | `run_threads.py` | Camera polling |
| `PhotonCountThread` | `run_threads.py` | Photon counter |

### Common Signals

| Signal | Emitter | Parameters | Purpose |
|--------|---------|------------|---------|
| `laser_power_changed` | ControllerPanel | `float` | Laser power in mW |
| `roi_updated` | ControllerPanel | `int, int, int, int` | x, y, width, height |
| `mode_changed` | ControllerPanel | `str` | Acquisition mode name |
| `frame_ready` | CameraAcquisitionThread | `np.ndarray` | New camera frame |
| `fft_computed` | FFTWorker | `np.ndarray` | FFT result |
| `image_reconstructed` | PSLiveWorker | `np.ndarray` | Reconstructed image |
| `acquisition_finished` | TaskWorker | None | Long task complete |

### Typical Parameter Ranges

| Parameter | Typical Range | Unit |
|-----------|---------------|------|
| **Camera ROI** | 512x512 to 2048x2048 | pixels |
| **Binning** | 1, 2, 4, 8 | - |
| **Gain** | 0-255 (EMCCD) | - |
| **Exposure** | 1-1000 | ms |
| **Laser Power** | 0-100 | mW |
| **Piezo Range** | 0-2.0 (X,Y), 0-0.0 (Z) | mm |
| **DAQ Sample Rate** | 1000-500000 | Hz |
| **Zernike Modes** | 4-36 (orders 2-6) | - |

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-01-22 | Initial CLAUDE.md creation - comprehensive documentation for AI assistants |

---

## Additional Resources

**External Documentation:**
- PyQt6: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- NumPy: https://numpy.org/doc/stable/
- National Instruments DAQmx: https://nidaqmx-python.readthedocs.io/

**Internal Documentation:**
- README.md: Project overview
- pyproject.toml: Dependencies and build configuration
- Configuration JSON examples: (not currently in repo - request from team)

**Getting Help:**
- Check logs: `~/Documents/data/YYYYMMDD_username/YYYYMMDD_HHMM.log`
- Review git history: `git log --oneline --graph --all`
- Examine recent commits on `point_scan` branch for latest features

---

*This document should be updated when significant architectural changes are made to the codebase.*
