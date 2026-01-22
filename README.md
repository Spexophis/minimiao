# MiniMiao

A Python GUI application for modular microscope control and imaging processing with Adaptive Optics.

## Overview

MiniMiao is a professional-grade microscopy control system that provides:

- **Multi-Device Support**: Control 16+ device types (cameras, lasers, SLMs, piezo stages, DAQs, deformable mirrors)
- **Advanced Imaging**: Wide-field, SIM 2D/3D, point scanning, photon counting
- **Real-time Visualization**: OpenGL-accelerated live imaging with FFT analysis
- **Adaptive Optics**: Wavefront sensing and correction
- **Precision Synchronization**: 250 kHz DAQ timing for hardware coordination

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

- **Python**: 3.10 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware** (optional): Supported microscopy devices (cameras, lasers, etc.)

### Install Python

If you don't have Python 3.10+, download it from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version  # Should show 3.10 or higher
```

### Install uv (Recommended Package Manager)

**uv** is a fast Python package installer and resolver. Install it:

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install via pip:
```bash
pip install uv
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

### 2. Install Dependencies

#### Option 1: Using uv (Recommended)

**uv** uses the locked dependencies in `uv.lock` for reproducible builds:

```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

#### Option 2: Using pip

If you prefer pip:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies
pip install -e .
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
   - If devices fail to connect, the app will gracefully fallback to mock devices
   - Check logs in `~/Documents/data/YYYYMMDD_username/YYYYMMDD_HHMM.log`

---

## Configuration

### Creating a Configuration File

Create a JSON configuration file with your hardware settings. Example:

```json
{
  "metadata": {
    "created": "2026-01-22",
    "description": "My MiniMiao Configuration"
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

See [CLAUDE.md](./CLAUDE.md#configuration-system) for complete configuration options.

### Testing Without Hardware

To test the application without physical devices:

1. Use the **MockCamera** (automatically used if EMCCD fails to connect)
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
├── src/minimiao/          # Main package
│   ├── __main__.py        # Entry point
│   ├── main.py            # AppWrapper
│   ├── executor.py        # CommandExecutor
│   ├── devices/           # Hardware drivers
│   ├── gui/               # PyQt6 GUI
│   ├── computations/      # Signal processing
│   └── utilities/         # Helper functions
├── pyproject.toml         # Project configuration
├── uv.lock                # Locked dependencies
└── CLAUDE.md              # Developer documentation
```

### Running Tests

Currently, the project uses manual testing with MockCamera. To test:

```bash
# Run with mock hardware (no devices needed)
python -m minimiao
```

### Code Style

- Follow PEP 8 conventions
- Use snake_case for functions/methods
- Use PascalCase for classes
- See [CLAUDE.md](./CLAUDE.md#code-conventions) for detailed conventions

---

## Troubleshooting

### Common Issues

#### 1. **Import Errors**

**Error**: `ModuleNotFoundError: No module named 'minimiao'`

**Solution**:
```bash
# Ensure you're in the project directory
cd minimiao

# Reinstall in editable mode
pip install -e .
```

#### 2. **PyQt6 Display Issues**

**Error**: `qt.qpa.plugin: Could not find the Qt platform plugin`

**Solution** (Linux):
```bash
# Install Qt dependencies
sudo apt-get install libxcb-xinerama0 libxcb-cursor0
```

#### 3. **Camera Not Detected**

**Issue**: EMCCD or other camera not connecting

**Solution**:
- Check device power and USB connection
- Verify device drivers are installed
- App will automatically fallback to MockCamera
- Check logs: `~/Documents/data/YYYYMMDD_username/YYYYMMDD_HHMM.log`

#### 4. **DAQ Errors**

**Error**: NIDAQmx errors or timing issues

**Solution**:
- Install NI-DAQmx drivers from National Instruments
- Verify device name in configuration JSON
- Check sample rate and channel assignments

#### 5. **Missing Dependencies**

**Error**: Import errors for specific packages

**Solution**:
```bash
# Regenerate lock file (uv)
uv lock

# Reinstall dependencies
uv sync

# Or with pip
pip install --upgrade -e .
```

### Enable Debug Logging

For detailed troubleshooting, enable debug logging:

1. Check logs in `~/Documents/data/YYYYMMDD_username/YYYYMMDD_HHMM.log`
2. All device operations and errors are logged in JSON format

### Getting Help

- **Issue Tracker**: [GitHub Issues](https://github.com/Spexophis/minimiao/issues)
- **Documentation**: See [CLAUDE.md](./CLAUDE.md) for comprehensive developer guide
- **Git History**: `git log --oneline --graph` for recent changes

---

## Documentation

- **[CLAUDE.md](./CLAUDE.md)**: Comprehensive developer documentation covering:
  - Architecture and components
  - Development workflows
  - Code conventions
  - Device management
  - Threading patterns
  - Common tasks and gotchas

---

## License

MIT License - Copyright (c) 2025 Spexophis

See [LICENSE](./LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Acknowledgments

This application is inspired by [ImSwitch](https://github.com/ImSwitch/ImSwitch) and follows the Model-View-Presenter (MVP) architecture.

Parts of the Python code, especially the adaptive optics processing, are inherited from the [SIM Control Software](https://github.com/Knerlab/SIM_Control_Software) repository.

**Built with:**
- PyQt6 for GUI framework
- NumPy/SciPy for scientific computing
- PyOpenGL for visualization
- NIDAQmx for hardware synchronization

---

**Happy Microscopy! 🔬**
