# Breathing Belt Python

`breathing-belt-python` is a publication-oriented artifact for acquiring a live breathing-belt signal, applying robust calibration and adaptive normalization, and exporting a reproducible session record.

The output is a **normalized breathing control signal** intended for real-time interaction, biofeedback, or downstream synchronization. It is **not** a validated respiratory-volume estimator and should not be described as a clinical or physiological volume measurement.

## Scope

This repository provides:
- BITalino acquisition through a threaded reader
- real-time high-pass plus low-pass filtering
- percentile-based startup calibration
- adaptive normalization with center and amplitude tracking
- raw-signal quality-control warnings and logging
- live plotting and optional LSL streaming
- automatic per-run export of raw rows, derived signals, QC events, and metadata

This repository does not provide:
- a validated tidal-volume estimator
- a medical device workflow
- cross-subject calibration or clinical interpretation

## Hardware and Environment

- BITalino device with belt sensor connected on the configured analog channels
- Python `3.11.4`
- Operating system support for the `keyboard` package if you use the live stop key

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -e .[dev]
```

## Configuration

Copy `config.example.toml` to `config.toml` and edit at least the device MAC address:

```bash
copy config.example.toml config.toml
```

Important config fields:
- `device.mac_address`: required BITalino MAC address
- `device.channels`: acquired analog channels
- `device.processed_sensor_column`: device-row column used for the normalized signal
- `filter.*`: real-time filter parameters
- `calibration.*`: processed-signal calibration settings
- `adaptation.*`: startup and runtime adaptive normalization settings
- `raw_qc.*`: raw-signal clipping, flatline, and baseline-shift thresholds
- `output.root_dir`: parent directory for timestamped session exports

## Running

Module entrypoint:

```bash
python -m src.main --config config.toml
```

Direct script entrypoint:

```bash
python src/main.py --config config.toml
```

Installed console script:

```bash
breathing-belt --config config.toml
```

During a live run:
- startup calibration is performed first
- the normalized signal is plotted if `display.enable_plot = true`
- LSL output is sent if `lsl.enable = true`
- acquisition stops when the `c` key is pressed

## Session Export

Each run creates a timestamped folder under `runs/` by default:

```text
runs/<timestamp>/
  resolved_config.toml
  session_metadata.json
  device_samples.csv
  signal_trace.csv
  qc_events.csv
```

File contents:
- `resolved_config.toml`: exact config used for the run
- `session_metadata.json`: session timestamps, software version, calibration result, final adaptive state, QC summary, and channel selection
- `device_samples.csv`: full device rows with `stage`, `sample_index`, and `relative_time_s`
- `signal_trace.csv`: processed signal values, normalized output, artifact flags, and hold-mode state
- `qc_events.csv`: one logged QC event per continuous clipping, flatline, or baseline-shift episode

The `stage` column distinguishes `calibration` from `runtime`.

## Signal-Processing Method

The live signal path is:

`raw selected channel -> high-pass filter -> low-pass filter -> sign inversion -> reversal-only artifact suppression -> calibration/adaptive normalization`

Calibration:
- runs on the processed signal
- uses percentile bounds (`percentile_lo`, `percentile_hi`)
- estimates a center and amplitude for normalization

Adaptive normalization:
- can update both center and amplitude over time
- uses faster adaptation immediately after calibration
- freezes adaptation during artifact-gated samples and detected breath holds

## Raw Quality Control

Raw QC runs on the selected raw sensor channel and warns/logs:
- saturation or clipping at configured low/high thresholds
- flatline behavior for a configured duration
- abrupt baseline shifts consistent with belt slip or posture changes

QC policy is advisory:
- warnings are printed during the run
- events are logged to `qc_events.csv`
- acquisition is not aborted automatically

## Limitations and Non-Claims

- The normalized output is a filtered breathing-control proxy, not a validated respiratory-volume estimate.
- Belt placement, posture, slack, and motion can materially affect the signal.
- The method currently processes one configured sensor column for normalization, even if multiple channels are acquired and exported.
- The code is intended for reproducible research workflows, not medical use.

## Tests

Run the repository checks with:

```bash
python -m compileall src tests
python -m pytest -q tests -p no:cacheprovider
```
