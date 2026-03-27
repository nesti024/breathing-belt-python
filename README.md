# Breathing Belt Python

`breathing-belt-python` is a publication-oriented artifact for acquiring a live breathing-belt signal, selecting a live processing mode at startup, and exporting a reproducible session record.

The runtime currently supports three live modes:
- `1`: a **normalized breathing control signal** for real-time interaction, biofeedback, or downstream synchronization
- `2`: a **realtime movement proxy** intended to stay closer to filtered belt motion in centered sensor units
- `3`: an **adaptive live control signal** that updates its operating range online to better follow changing breathing rhythms

None of these modes is a validated respiratory-volume estimator, and none should be described as a clinical or physiological volume measurement.

## Scope

This repository provides:
- BITalino acquisition through a threaded reader
- startup selection between legacy control, movement-proxy, and adaptive live-control modes
- percentile-based startup calibration
- fixed-calibration mapping with padded control headroom into a continuous `0..1` breath level in control mode
- centered movement-proxy output with causal drift removal and light smoothing in movement mode
- adaptive center/amplitude normalization for rhythm-flexible live control
- inhale-peak and exhale-trough event detection in all live modes
- breath-hold output freezing near inhale/exhale extremes to minimize sphere drift during holds in control mode
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
- `filter.lp_*`: low-pass parameters for legacy control mode and adaptive live mode
- `movement.*`: high-pass and low-pass parameters for realtime movement-proxy mode
- `calibration.*`: processed-signal calibration settings, including control-map headroom via `padding_ratio`
- `adaptation.*`: runtime center/amplitude update speeds for adaptive live mode
- `hold.*`: breath-hold freeze thresholds and the extrema-zone gate via `edge_margin_ratio`; set `hold.enabled = false` to disable hold detection for testing
- `output_smoothing.*`: motion-adaptive damping for the emitted `0..1` control signal, including faster convergence near real extremes via `tau_extreme_s` and `edge_margin_ratio`
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
- choose `1`, `2`, or `3` at the startup console prompt
- startup calibration is performed first
- the raw signal is plotted during calibration and runtime if `display.enable_plot = true`
- the mode-specific secondary signal is plotted after calibration
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

At startup the user selects one of three live modes.

Mode `1` (`Legacy control`):

`raw selected channel -> optional polarity inversion -> low-pass filter -> fixed startup calibration -> padded control bounds -> optional hold gate -> motion-adaptive output smoothing -> emitted 0..1 breath level`

Mode `2` (`Realtime movement proxy`):

`raw selected channel -> optional polarity inversion -> causal high-pass drift removal -> causal low-pass smoothing -> startup calibration -> centered movement output`

Mode `3` (`Adaptive live control`):

`raw selected channel -> optional polarity inversion -> low-pass filter -> startup calibration -> adaptive center/amplitude update -> emitted 0..1 breath level`

Calibration:
- runs on the processed signal in all modes
- uses percentile bounds (`percentile_lo`, `percentile_hi`)
- estimates a fixed control map for mode `1`
- estimates a fixed center and reference amplitude for mode `2`
- seeds the initial adaptive operating range for mode `3`

Runtime control mode:
- uses the fixed startup calibration for the entire run
- maps the filtered signal through the padded control bounds so full breaths do not clip as early
- optionally freezes the emitted `0..1` value during low-activity breath holds near the top or bottom `edge_margin_ratio` of the range
- releases freeze immediately when motion resumes or the live control value drifts away from the frozen value
- applies motion-adaptive smoothing to the final emitted `0..1` value: fast when breathing is active, slow when activity is low, and faster again near the bottom/top `edge_margin_ratio` of the range so real extremes remain reachable
- emits `1.0` on inhale peaks and `-1.0` on exhale troughs

Runtime movement mode:
- outputs centered filtered sensor units rather than a normalized `0..1` control level
- does not apply padded control mapping, hold freezing, or output smoothing
- still emits `1.0` on inhale peaks and `-1.0` on exhale troughs

Runtime adaptive mode:
- emits a bounded `0..1` control level while updating center and amplitude online
- uses `adaptation.startup_*` time constants during the startup adaptation window and `adaptation.center_tau_s` / `adaptation.amplitude_tau_s` afterward
- does not apply hold freezing or motion-adaptive output smoothing
- also exports a centered `movement_value` trace derived from the current adaptive center
- still emits `1.0` on inhale peaks and `-1.0` on exhale troughs

By default, the local `config.toml` keeps `hold.enabled = false` and relies on `output_smoothing` as the primary anti-drift mechanism for testing.

LSL:
- mode `1` publishes two float32 channels by default: `breath_level` and `event_code`
- mode `2` publishes a separate stream identity with `movement_value` and `event_code`
- mode `3` publishes a separate stream identity with `breath_level` and `event_code`
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

- The normalized control output, adaptive live output, and movement-proxy output are all filtered proxies, not validated respiratory-volume estimates.
- Belt placement, posture, slack, and motion can materially affect the signal.
- The method currently processes one configured sensor column for normalization, even if multiple channels are acquired and exported.
- The deprecated `filter.hp_*` and `artifact.*` settings remain loadable for compatibility and are not used by the fixed-calibration VR control path.
- The code is intended for reproducible research workflows, not medical use.

## Tests

Run the repository checks with:

```bash
python -m compileall src tests
python -m pytest -q tests -p no:cacheprovider
```
