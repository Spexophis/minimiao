# MiniMiao
A Python application for modular microscope control and imaging processing with Adaptive Optics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Prerequisites

### System Requirements

- **Python**: Python: 3.9–3.11 (3.12+ compatibility depends on hardware interface)
- **Operating System**: Windows

### Install Python

Python can be downloaded from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
```

### Install uv (Recommended Package Manager)

**uv** is a fast Python package installer and resolver: https://docs.astral.sh/uv/#highlights

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install via pip:
```bash
pip install uv
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
```

---

## Installation

### 1. Clone the Repository

```bash
# Clone via HTTPS
git clone https://github.com/Spexophis/minimiao.git

# OR clone via SSH
git clone git@github.com:Spexophis/minimiao.git

# Navigate to project directory
cd minimiao
```

### 2. Install Dependencies using uv (Recommended)

**uv** uses the locked dependencies in `uv.lock` for reproducible builds:

```bash
# Create virtual environment and install dependencies
uv sync

# Windows:
.venv\Scripts\activate

# Activate the virtual environment
# Linux/macOS:
source .venv/bin/activate
```

### 3. Verify Installation

Check that minimiao is installed:

```bash
python -c "import minimiao; print('MiniMiao installed successfully!')"
```

---

## Running the Application

### Standard Run

```bash
# Ensure virtual environment is activated
python -m minimiao
```

### Run with Module Import

```python
# In Python interpreter or script
from minimiao import main

app_wrapper = main.AppWrapper()
app_wrapper.run()
```

### First-Time Setup

When you run MiniMiao for the first time:

1. **Configuration File**: A file dialog will prompt you to select a JSON configuration file
2. **Data Directory**: The app will create `~/Documents/data/YYYYMMDD_username/` for logs and data
3. **Device Initialization**: The app will attempt to connect to configured hardware
   - If devices fail to connect, check logs in `~/Documents/data/YYYYMMDD_username/YYYYMMDD_HHMM.log`

---

## Configuration

### Creating a Configuration File

Create a JSON configuration file with your hardware settings. Example:

```json
{
    "Data Path": "C:\\Users\\Public\\Documents\\data",
    "ConWidget Path": "C:\\Users\\Public\\Documents\\data\\config_files\\conwidget_values_slm_parallel_scan.json",
    "AOWidget Path": "C:\\Users\\Public\\Documents\\data\\config_files\\aowidget_values_slm_parallel_scan.json",
    "Digital Timing Presets": "C:\\Users\\Public\\Documents\\data\\config_files\\digital_timing_presets_slm_parallel_scan.json",
    "Cameras": {
        "Andor EMCCD": {
            "Model": "iXon Life 888",
            "Serial": "X-13693",
            "Pixel Size": 13,
            "Unit": "um",
            "Pixel Number Horizontal": 1024,
            "Pixel Number Vertical": 1024
        },
        "Hamamatsu sCMOS": {
            "Model": "ORCA Flash 4.0 C11440-22CU",
            "Serial": "100511",
            "Pixel Size": 6.5,
            "Unit": "um",
            "Pixel Number Horizontal": 2048,
            "Pixel Number Vertical": 2048
        },
        "TIS CMOS": {
            "Model": "DMK 33UX250",
            "Serial": "10811087",
            "Pixel Size": 3.45,
            "Unit": "um",
            "Pixel Number Horizontal": 2448,
            "Pixel Number Vertical": 2048
        }
    },
    "Triggers": {
        "NIDAQ": {
            "Dev1": {
                "Model": "PCIe-6353",
                "Serial": "",
                "Sampling Rate": 2500000,
                "Unit": "Hz"
            },
            "Dev2": {
                "Model": "PCIe-6353",
                "Serial": "",
                "Sampling Rate": 2500000,
                "Unit": "Hz"
            },
            "Channels": {
                "analog_output_channels": {
                    "piezo_x": "Dev1/ao0",
                    "piezo_y": "Dev1/ao1",
                    "piezo_z": "Dev1/ao2",
                    "galvo_swx": "Dev2/ao0",
                    "galvo_swy": "Dev2/ao1"
                },
                "digital_output_channels": {
                    "laser_405": "Dev1/port0/line0",
                    "laser_488_w": "Dev1/port0/line1",
                    "laser_488": "Dev1/port0/line3",
                    "andor ccd": "Dev1/port0/line4",
                    "hamamatsu scmos": "Dev1/port0/line5",
                    "tis cmos": "Dev1/port0/line7"
                },
                "analog_input_channels": {
                    "piezo_x": "Dev1/ai0",
                    "piezo_y": "Dev1/ai1",
                    "piezo_z": "Dev1/ai2",
                    "galvo_swx": "Dev2/ai0",
                    "galvo_swy": "Dev2/ai1"
                }
            }
        }
    },
    "Light Sources": {
        "Lasers": {
            "Cobolt": {
                "405": {
                    "Model": "0405-06-01-0250-100",
                    "Serial": "11735",
                    "Wavelength": "405 nm"
                },
                "488_w": {
                    "Model": "0488-06-01-0200-100",
                    "Serial": "12077",
                    "Wavelength": "488 nm"
                },
                "488": {
                    "Model": "0488-06-01-0200-100",
                    "Serial": "24292",
                    "Wavelength": "488 nm"
                }
            }
        }
    },
    "Sample Stages": {
        "MCL Piezo Stage": {
            "Model": "Nano-LP100",
            "Serial": "",
            "Translation XY": 100,
            "Translation Z": 100,
            "Unit": "um"
        },
        "MCL Mad-Deck": {
            "Model": "Mad-Deck",
            "Serial": "",
            "Translation XY": 0,
            "Translation Z": 23,
            "Translation Unit": "mm",
            "Step Precision": 95.25,
            "Precision Unit": "nm"
        }
    },
    "Beam Steerers": {
        "Galvo Mirrors": {
            "ScannerMax": {
                "Model": "Saturn 9B XY System",
                "Serial": "PS904561"
            }
        }
    },
    "Spatial Light Modulator": {
        "Forth Dimension Displays": {
            "Model": "M150 QXGA",
            "Serial": "175000787",
            "ControlLibrary": "C:\\Program Files\\MetroCon-4.2\\lib\\NativeLibs\\R11CommLib-x64.dll",
            "Pixel Pitch": 8.3,
            "Unit": "um",
            "Pixel Number Horizontal": 2048,
            "Pixel Number Vertical": 1536
        }
    },
    "Adaptive Optics": {
        "Deformable Mirrors": {
            "ALPAO DM97": {
                "Model": "DM97-15",
                "Serial": "BAX513",
                "Actuator Number": 97,
                "Pitch Size": 1.5,
                "Pupil Diameter": 13.5,
                "Unit": "mm",
                "Calibration File Folder": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513",
                "Initial Flat": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513\\flat_file_BAX513_2025_08_29_11_24.xlsx",
                "Phase Control Matrix": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513\\control_matrix_phase_2025_09_08_17_31.tif",
                "Zonal Control Matrix": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513\\control_matrix_zonal_2025_09_08_17_31.tif",
                "Modal Control Matrix": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513\\control_matrix_modal_2025_09_08_17_31.tif",
                "Influence Function Images": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513\\influence_function_images_2025_09_08_17_31.tif",
                "Control Calibration": "C:\\Users\\Public\\Documents\\data\\dm_files\\bax513\\control_calibration_20240628.npz"
            }
        }
    }
}
```

### Testing Without Hardware

To test the application without physical devices:

1. Use the **MockCamera** (automatically used if physical camera fails to connect)
2. Comment out device configurations in your JSON file
3. The GUI will remain functional for testing workflows

---

## Development Setup

### Install in Development Mode

For development with editable installation:

```bash
# Using uv
uv pip install -e .

# Using pip
pip install -e .
```

### Project Structure

```
minimiao/
├── src/minimiao/          
│   ├── __main__.py        
│   ├── main.py            
│   ├── gui/               # PyQt6 GUI
│   ├── devices/           # Hardware interface wrappers / APIs
│   ├── executor.py        # Command executor & resource manager
│   ├── run_threads.py     # Multithreading infrastructure
│   ├── computations/      # Image processing & signal computation
│   └── utilities/         # Shared utility functions
├── pyproject.toml         
└── uv.lock                # Locked dependencies
```

### Running Tests

Currently, the project uses manual testing with MockCamera. To test:

```bash
# Run with mock hardware (no devices needed)
python -m minimiao
```

## Acknowledgments

This application is inspired by [ImSwitch](https://github.com/ImSwitch/ImSwitch) and follows the Model-View-Presenter (MVP) architecture.
 
The adaptive optics components are developed based on the [SIM Control Software](https://github.com/Knerlab/SIM_Control_Software).
 
---
