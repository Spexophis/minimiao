# MiniMiao
A Python application for modular microscope control and imaging processing with Adaptive Optics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option 1: Using uv (Recommended)](#option-1-using-uv-recommended)
  - [Option 2: Using pip](#option-2-using-pip)
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
  "metadata": {
    "created": "2026-01-22",
    "description": "MiniMiao Configuration"
  },

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

  "daq": {
    "type": "ni_daq",
    "device_name": "Dev1",
    "sample_rate": 250000
  },

  "acquisition": {
    "data_directory": "~/Documents/data",
    "default_mode": "wide_field"
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
