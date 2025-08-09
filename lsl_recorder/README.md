# LSL Stream Recorder for OpenBCI

A set of Python scripts for recording LSL (Lab Streaming Layer) data streams from OpenBCI hardware, with full timestamping and data upload capabilities.

## Features

- **Multi-stream recording** - Capture multiple LSL streams simultaneously
- **Analysis streams** - Specialized recording for TimeSeriesFilt, BandPower, and FFT streams
- **Full timestamping** - Each sample includes unix timestamp, LSL timestamp, and ISO format
- **Automatic data saving** - All streams saved to CSV files in `lsl_data/` directory
- **Upload capability** - Script to upload recorded data to remote servers

## Files

### 1. `openbci_lsl_reader.py`
Basic single-stream LSL reader with file saving capability.

### 2. `openbci_multi_stream_reader.py`
Records ALL available LSL streams simultaneously in parallel threads.

### 3. `openbci_analysis_streams_reader.py`
Specialized recorder for the 3 main analysis streams:
- **TimeSeriesFilt** - Filtered EEG time series data
- **BandPower** - Power in frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **FFT** - Frequency spectrum data

### 4. `upload_lsl_data.py`
Uploads recorded CSV files to a server with metadata extraction.

## Installation

```bash
pip install pylsl numpy
```

## Usage

### Record Analysis Streams (Recommended)
```bash
python3 openbci_analysis_streams_reader.py
```
This will automatically find and record TimeSeriesFilt, BandPower, and FFT streams.

### Record All Available Streams
```bash
python3 openbci_multi_stream_reader.py
```

### Upload Recorded Data
```bash
python3 upload_lsl_data.py --status  # Check local files
python3 upload_lsl_data.py --all     # Upload all files
python3 upload_lsl_data.py --latest  # Upload most recent
```

## Data Format

All CSV files include:
- Metadata headers with stream information
- Three timestamp formats per sample:
  - `unix_timestamp` - System time (for easy processing)
  - `lsl_timestamp` - LSL protocol timestamp
  - `system_time_iso` - Human-readable timestamp
- Channel data

Example structure:
```csv
# OpenBCI Analysis Stream Data
# Stream Type: BANDPOWER
# Session Start: 2024-12-09T14:30:22
# Stream Name: BandPower
# Sampling Rate: 10.0 Hz
# Channel Count: 5
unix_timestamp, lsl_timestamp, system_time_iso, delta, theta, alpha, beta, gamma
1702134622.123, 45678.901, 2024-12-09T14:30:22.123456, 0.5, 0.3, 0.8, 0.4, 0.2
```

## Directory Structure
```
lsl_data/
├── timeseriesfilt_20241209_143022.csv
├── bandpower_20241209_143022.csv
├── fft_20241209_143022.csv
└── ...
```

## Requirements

- Python 3.6+
- pylsl library
- OpenBCI GUI with LSL streaming enabled
- Enable required widgets in OpenBCI GUI:
  - Time Series widget (for filtered data)
  - Band Power widget
  - FFT widget

## Notes

- All streams from the same session share the same timestamp suffix for easy correlation
- Data is saved continuously as it arrives (no data loss on interruption)
- Supports both timed and continuous recording modes
- Thread-safe parallel recording for multiple streams