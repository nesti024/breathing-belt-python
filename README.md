# Breathing Belt Python

`breathing-belt-python` is a publication-oriented artifact for acquiring a live breathing-belt signal, applying fixed-calibration breathing control mapping with extrema events, and exporting a reproducible session record.

The output is a **normalized breathing control signal** intended for real-time interaction, biofeedback, or downstream synchronization. It is **not** a validated respiratory-volume estimator and should not be described as a clinical or physiological volume measurement.

## Scope

This repository provides:
- BITalino acquisition through a threaded reader
- real-time low-pass filtering for VR-oriented breathing control
- percentile-based startup calibration
- fixed-calibration mapping into a continuous `0..1` breath level
- inhale-peak and exhale-trough event detection
- breath-hold output freezing to minimize sphere drift during holds
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
- `device.invert_signal`: flips the control-signal polarity when inhale/exhale direction is reversed
- `filter.lp_*`: low-pass control filter parameters
- `calibration.*`: processed-signal calibration settings
- `hold.*`: breath-hold freeze thresholds for the output level
- `extrema.*`: minimum interval and prominence thresholds for inhale/exhale events
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
- the raw signal is plotted during calibration and runtime if `display.enable_plot = true`
- the `0..1` breath level is plotted after calibration
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
- `session_metadata.json`: session timestamps, software version, fixed calibration details, stream metadata, QC summary, and channel selection
- `device_samples.csv`: full device rows with `stage`, `sample_index`, and `relative_time_s`
- `signal_trace.csv`: filtered control values, normalized output, hold/freeze state, and inhale/exhale event codes
- `qc_events.csv`: one logged QC event per continuous clipping, flatline, or baseline-shift episode

The `stage` column distinguishes `calibration` from `runtime`.

## Signal-Processing Method

The live control path is:

`raw selected channel -> optional polarity inversion -> low-pass filter -> fixed startup calibration -> clamped 0..1 breath level`

Calibration:
- runs on the processed signal
- uses percentile bounds (`percentile_lo`, `percentile_hi`)
- estimates a fixed center and amplitude for the runtime control map

Runtime control:
- uses the fixed startup calibration for the entire run
- freezes the emitted `0..1` value during low-activity breath holds
- emits `1.0` on inhale peaks and `-1.0` on exhale troughs

LSL:
- publishes two float32 channels by default: `breath_level` and `event_code`
- `event_code` is `0.0` during normal samples, `1.0` for inhale peaks, and `-1.0` for exhale troughs

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
- The deprecated `filter.hp_*`, `artifact.*`, and `adaptation.*` settings remain loadable for compatibility but are not used by the fixed-calibration VR control path.
- The code is intended for reproducible research workflows, not medical use.

## Tests

Run the repository checks with:

```bash
python -m compileall src tests
python -m pytest -q tests -p no:cacheprovider
```
