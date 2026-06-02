# MP-QUIC AI IoT System Setup Guide

This guide covers the installation and environment setup for the MP-QUIC AI prediction system across different operating systems and deployment targets (Server, Raspberry Pi, and Development Machines).

## 1. System-Level Dependencies

Before installing Python packages, you must install system-level build dependencies. The transport layer relies on `aioquic`, which requires a C compiler and OpenSSL development headers to build its cryptography and QPACK extensions.

### Linux (Debian / Ubuntu / Raspberry Pi OS)
This is the primary target environment.
```bash
sudo apt-get update
sudo apt-get install python3-dev libssl-dev build-essential
```

### macOS (Development)
Use Homebrew to install OpenSSL.
```bash
brew install openssl
```
*Note: When installing `aioquic` later via pip, you may need to specify the OpenSSL path if the compiler cannot find it:*
```bash
CFLAGS="-I$(brew --prefix openssl)/include" LDFLAGS="-L$(brew --prefix openssl)/lib" pip install aioquic==1.3.0
```

### Windows (Development)
Native compilation of `aioquic` on Windows can be complex due to OpenSSL and compiler toolchain requirements.
**Recommendation:** Use **WSL2** (Windows Subsystem for Linux - Ubuntu) for development on Windows, and follow the Linux instructions above.

---

## 2. Installation Instructions

Follow these steps to set up the project environment.

### Step 1: Create a Virtual Environment
It is highly recommended to isolate project dependencies using a virtual environment.

```bash
# Create the virtual environment (requires Python 3.10+)
python3 -m venv .venv

# Activate the virtual environment
# On Linux / macOS / WSL:
source .venv/bin/activate
# On Windows (if not using WSL):
# .\.venv\Scripts\activate
```

### Step 2: Install System Dependencies
Ensure you have installed the system dependencies listed in Section 1 for your respective OS.

### Step 3: Install Python Dependencies
Install the required packages from `requirements.txt`. 

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```
> **Important Note for Development Machines:** The `requirements.txt` contains a section at the bottom for Raspberry Pi-specific hardware libraries (`adafruit-circuitpython-dht`, `RPi.GPIO`). These are **commented out** by default. Only uncomment and install them when deploying directly to the Raspberry Pi hardware.

### Step 4: Configure Environment Variables
The system relies on environment variables for configuration (IP addresses, ports, thresholds).

1. Create a file named `.env` in the root of the project.
2. Add the following default variables (adjust values as needed for your specific network/setup):

```env
# ── Server connectivity ────────────────────────────────────────────────────────
SERVER_HOST=127.0.0.1
SERVER_PORT_PATH1=5001
SERVER_PORT_PATH2=5002

# ── Allowlisted client IPs ────────────────────────────────────────────────────
# Comma-separated list of IP addresses permitted to connect to mpquic_server.
# In production, set this to the Raspberry Pi's LAN IP.
# During development/emulation, 127.0.0.1 is sufficient.
ALLOWED_CLIENT_IPS=127.0.0.1

# ── Timing ────────────────────────────────────────────────────────────────────
SEND_INTERVAL_SEC=0.5
SENSOR_INTERVAL_SEC=2.0

# ── Thresholds ────────────────────────────────────────────────────────────────
RTT_DEGRADATION_MS=100.0

# ── Network interfaces (Raspberry Pi only) ────────────────────────────────────
PATH1_IFACE=wlan0
PATH2_IFACE=eth0
```

---

## 3. Verifying the Installation

To verify that the core libraries (specifically the ML stack and the QUIC transport layer) have been installed correctly, run the following command within your activated virtual environment:

```bash
python3 -c "
import aioquic
import tensorflow as tf
import fastapi
import sqlalchemy
print('\n✅ Installation verified successfully!')
print(f'   aioquic version:    {aioquic.__version__}')
print(f'   tensorflow version: {tf.__version__}')
print(f'   fastapi version:    {fastapi.__version__}')
print(f'   sqlalchemy version: {sqlalchemy.__version__}')
"
```

If the script outputs the version numbers without throwing any `ModuleNotFoundError` or compilation-related exceptions, your environment is correctly configured and ready to run the MP-QUIC server and ML pipelines.
