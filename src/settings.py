"""Configuration loading and serialization for breathing-belt runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class DeviceConfig:
    """Hardware and acquisition settings."""

    mac_address: str = ""
    sampling_rate_hz: int = 100
    channels: tuple[int, ...] = (0, 1)
    processed_sensor_column: int = 5
    chunk_size: int = 10
    queue_max_samples: int = 1000
    timeout_s: float = 0.25
    read_error_backoff_s: float = 0.05
    retries: int = 3
    retry_delay_s: float = 2.0
    invert_signal: bool = False


@dataclass(frozen=True)
class DisplayConfig:
    """Live plotting and terminal-debug settings."""

    enable_plot: bool = True
    plot_window_length: int = 3000
    debug_plot_window_bounds: bool = True


@dataclass(frozen=True)
class LSLConfig:
    """Lab Streaming Layer output settings."""

    enable: bool = True
    stream_name: str = "BreathingBelt"
    stream_type: str = "Breathing"
    source_id: str = "breathingbelt001"
    constant_delay_s: float = 0.0


@dataclass(frozen=True)
class FilterConfig:
    """Causal filter parameters for the live breathing signal."""

    hp_cutoff_hz: float = 0.005
    hp_order: int = 1
    lp_cutoff_hz: float = 1.5
    lp_order: int = 2


@dataclass(frozen=True)
class MovementConfig:
    """Causal filter parameters for the live movement-proxy signal."""

    hp_cutoff_hz: float = 0.03
    hp_order: int = 1
    lp_cutoff_hz: float = 1.5
    lp_order: int = 2
    low_activity_slowdown_enabled: bool = False
    low_activity_window_ms: int = 800
    low_activity_ratio_per_sec: float = 0.05
    low_activity_floor_per_sec: float = 0.01
    low_activity_drift_scale: float = 0.15


@dataclass(frozen=True)
class CalibrationSettings:
    """Percentile calibration settings and control-map headroom."""

    duration_s: float = 15.0
    percentile_lo: float = 5.0
    percentile_hi: float = 95.0
    amplitude_floor: float = 1e-3
    padding_ratio: float = 0.20


@dataclass(frozen=True)
class AdaptationSettings:
    """Adaptive normalization settings."""

    center_enabled: bool = True
    amplitude_enabled: bool = True
    center_tau_s: float = 600.0
    amplitude_tau_s: float = 120.0
    startup_duration_s: float = 60.0
    startup_center_tau_s: float = 25.0
    startup_amplitude_tau_s: float = 25.0
    low_activity_gating_enabled: bool = True
    low_activity_window_ms: int = 800
    low_activity_ratio_per_sec: float = 0.05
    low_activity_floor_per_sec: float = 0.01


@dataclass(frozen=True)
class HoldConfig:
    """Breath-hold freeze thresholds and extrema-zone gating."""

    enabled: bool = True
    activity_window_ms: int = 500
    ratio_per_sec_enter: float = 1.0
    ratio_per_sec_exit: float = 1.5
    floor_per_sec: float = 0.01
    edge_margin_ratio: float = 0.20


@dataclass(frozen=True)
class OutputSmoothingConfig:
    """Motion-adaptive smoothing for the emitted 0..1 control output."""

    enabled: bool = True
    activity_window_ms: int = 500
    tau_active_s: float = 0.25
    tau_extreme_s: float = 0.75
    tau_hold_s: float = 5.0
    activity_low_ratio_per_sec: float = 0.10
    activity_high_ratio_per_sec: float = 0.50
    activity_floor_per_sec: float = 0.01
    edge_margin_ratio: float = 0.10


@dataclass(frozen=True)
class ExtremaConfig:
    """Peak/trough confirmation parameters for breathing events."""

    min_interval_ms: int = 800
    prominence_ratio: float = 0.1


@dataclass(frozen=True)
class RawQCConfig:
    """Raw-signal quality-control settings."""

    enabled: bool = True
    raw_saturation_lo: float = 1.0
    raw_saturation_hi: float = 1022.0
    flatline_epsilon: float = 0.5
    flatline_duration_s: float = 2.0
    baseline_ema_tau_s: float = 30.0
    baseline_abs_dev_tau_s: float = 10.0
    baseline_shift_sigma: float = 8.0
    baseline_shift_floor: float = 50.0
    warmup_s: float = 5.0


@dataclass(frozen=True)
class OutputConfig:
    """Per-run export settings."""

    root_dir: str = "runs"


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    device: DeviceConfig
    display: DisplayConfig
    lsl: LSLConfig
    filter: FilterConfig
    movement: MovementConfig
    calibration: CalibrationSettings
    adaptation: AdaptationSettings
    hold: HoldConfig
    output_smoothing: OutputSmoothingConfig
    extrema: ExtremaConfig
    raw_qc: RawQCConfig
    output: OutputConfig


def default_config() -> AppConfig:
    """Return the default configuration template."""

    return AppConfig(
        device=DeviceConfig(),
        display=DisplayConfig(),
        lsl=LSLConfig(),
        filter=FilterConfig(),
        movement=MovementConfig(),
        calibration=CalibrationSettings(),
        adaptation=AdaptationSettings(),
        hold=HoldConfig(),
        output_smoothing=OutputSmoothingConfig(),
        extrema=ExtremaConfig(),
        raw_qc=RawQCConfig(),
        output=OutputConfig(),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load a TOML configuration file into typed application settings.

    Missing config files fall back to the built-in defaults so tooling and dry
    runs can resolve a complete configuration without device-specific setup.
    """

    config_path = Path(path)
    if not config_path.exists():
        config = default_config()
    else:
        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)

        config = AppConfig(
            device=_load_device_config(_section(raw_config, "device")),
            display=_load_display_config(_section(raw_config, "display")),
            lsl=_load_lsl_config(_section(raw_config, "lsl")),
            filter=_load_filter_config(_section(raw_config, "filter")),
            movement=_load_movement_config(_section(raw_config, "movement")),
            calibration=_load_calibration_settings(_section(raw_config, "calibration")),
            adaptation=_load_adaptation_settings(_section(raw_config, "adaptation")),
            hold=_load_hold_config(_section(raw_config, "hold")),
            output_smoothing=_load_output_smoothing_config(
                _section(raw_config, "output_smoothing")
            ),
            extrema=_load_extrema_config(_section(raw_config, "extrema")),
            raw_qc=_load_raw_qc_config(_section(raw_config, "raw_qc")),
            output=_load_output_config(_section(raw_config, "output")),
        )
    _validate_config(config)
    return config


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert a typed configuration object to a serializable nested mapping."""

    return asdict(config)


def write_config_toml(path: str | Path, config: AppConfig) -> None:
    """Write a resolved configuration object to TOML."""

    Path(path).write_text(render_toml(config_to_dict(config)) + "\n", encoding="utf-8")


def render_toml(mapping: dict[str, Any]) -> str:
    """Serialize a nested mapping containing scalars, arrays, and subtables to TOML."""

    lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.extend(_render_toml_section(key, value))
            lines.append("")
        else:
            lines.append(f"{key} = {_render_toml_value(value)}")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _render_toml_section(prefix: str, mapping: dict[str, Any]) -> list[str]:
    lines = [f"[{prefix}]"]
    nested_items: list[tuple[str, dict[str, Any]]] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            nested_items.append((f"{prefix}.{key}", value))
        else:
            lines.append(f"{key} = {_render_toml_value(value)}")
    for nested_key, nested_value in nested_items:
        lines.append("")
        lines.extend(_render_toml_section(nested_key, nested_value))
    return lines


def _render_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _section(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw_config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a table.")
    return section


def _load_device_config(section: dict[str, Any]) -> DeviceConfig:
    defaults = DeviceConfig()
    return DeviceConfig(
        mac_address=str(section.get("mac_address", defaults.mac_address)),
        sampling_rate_hz=int(section.get("sampling_rate_hz", defaults.sampling_rate_hz)),
        channels=tuple(int(x) for x in section.get("channels", defaults.channels)),
        processed_sensor_column=int(
            section.get("processed_sensor_column", defaults.processed_sensor_column)
        ),
        chunk_size=int(section.get("chunk_size", defaults.chunk_size)),
        queue_max_samples=int(section.get("queue_max_samples", defaults.queue_max_samples)),
        timeout_s=float(section.get("timeout_s", defaults.timeout_s)),
        read_error_backoff_s=float(
            section.get("read_error_backoff_s", defaults.read_error_backoff_s)
        ),
        retries=int(section.get("retries", defaults.retries)),
        retry_delay_s=float(section.get("retry_delay_s", defaults.retry_delay_s)),
        invert_signal=bool(section.get("invert_signal", defaults.invert_signal)),
    )


def _load_display_config(section: dict[str, Any]) -> DisplayConfig:
    defaults = DisplayConfig()
    return DisplayConfig(
        enable_plot=bool(section.get("enable_plot", defaults.enable_plot)),
        plot_window_length=int(
            section.get("plot_window_length", defaults.plot_window_length)
        ),
        debug_plot_window_bounds=bool(
            section.get(
                "debug_plot_window_bounds",
                defaults.debug_plot_window_bounds,
            )
        ),
    )


def _load_lsl_config(section: dict[str, Any]) -> LSLConfig:
    defaults = LSLConfig()
    return LSLConfig(
        enable=bool(section.get("enable", defaults.enable)),
        stream_name=str(section.get("stream_name", defaults.stream_name)),
        stream_type=str(section.get("stream_type", defaults.stream_type)),
        source_id=str(section.get("source_id", defaults.source_id)),
        constant_delay_s=float(
            section.get("constant_delay_s", defaults.constant_delay_s)
        ),
    )


def _load_filter_config(section: dict[str, Any]) -> FilterConfig:
    defaults = FilterConfig()
    return FilterConfig(
        hp_cutoff_hz=float(section.get("hp_cutoff_hz", defaults.hp_cutoff_hz)),
        hp_order=int(section.get("hp_order", defaults.hp_order)),
        lp_cutoff_hz=float(section.get("lp_cutoff_hz", defaults.lp_cutoff_hz)),
        lp_order=int(section.get("lp_order", defaults.lp_order)),
    )


def _load_movement_config(section: dict[str, Any]) -> MovementConfig:
    defaults = MovementConfig()
    return MovementConfig(
        hp_cutoff_hz=float(section.get("hp_cutoff_hz", defaults.hp_cutoff_hz)),
        hp_order=int(section.get("hp_order", defaults.hp_order)),
        lp_cutoff_hz=float(section.get("lp_cutoff_hz", defaults.lp_cutoff_hz)),
        lp_order=int(section.get("lp_order", defaults.lp_order)),
        low_activity_slowdown_enabled=bool(
            section.get(
                "low_activity_slowdown_enabled",
                defaults.low_activity_slowdown_enabled,
            )
        ),
        low_activity_window_ms=int(
            section.get(
                "low_activity_window_ms",
                defaults.low_activity_window_ms,
            )
        ),
        low_activity_ratio_per_sec=float(
            section.get(
                "low_activity_ratio_per_sec",
                defaults.low_activity_ratio_per_sec,
            )
        ),
        low_activity_floor_per_sec=float(
            section.get(
                "low_activity_floor_per_sec",
                defaults.low_activity_floor_per_sec,
            )
        ),
        low_activity_drift_scale=float(
            section.get(
                "low_activity_drift_scale",
                defaults.low_activity_drift_scale,
            )
        ),
    )
def _load_calibration_settings(section: dict[str, Any]) -> CalibrationSettings:
    defaults = CalibrationSettings()
    return CalibrationSettings(
        duration_s=float(section.get("duration_s", defaults.duration_s)),
        percentile_lo=float(section.get("percentile_lo", defaults.percentile_lo)),
        percentile_hi=float(section.get("percentile_hi", defaults.percentile_hi)),
        amplitude_floor=float(
            section.get("amplitude_floor", defaults.amplitude_floor)
        ),
        padding_ratio=float(section.get("padding_ratio", defaults.padding_ratio)),
    )


def _load_adaptation_settings(section: dict[str, Any]) -> AdaptationSettings:
    defaults = AdaptationSettings()
    return AdaptationSettings(
        center_enabled=bool(section.get("center_enabled", defaults.center_enabled)),
        amplitude_enabled=bool(
            section.get("amplitude_enabled", defaults.amplitude_enabled)
        ),
        center_tau_s=float(section.get("center_tau_s", defaults.center_tau_s)),
        amplitude_tau_s=float(section.get("amplitude_tau_s", defaults.amplitude_tau_s)),
        startup_duration_s=float(
            section.get("startup_duration_s", defaults.startup_duration_s)
        ),
        startup_center_tau_s=float(
            section.get("startup_center_tau_s", defaults.startup_center_tau_s)
        ),
        startup_amplitude_tau_s=float(
            section.get(
                "startup_amplitude_tau_s",
                defaults.startup_amplitude_tau_s,
            )
        ),
        low_activity_gating_enabled=bool(
            section.get(
                "low_activity_gating_enabled",
                defaults.low_activity_gating_enabled,
            )
        ),
        low_activity_window_ms=int(
            section.get(
                "low_activity_window_ms",
                defaults.low_activity_window_ms,
            )
        ),
        low_activity_ratio_per_sec=float(
            section.get(
                "low_activity_ratio_per_sec",
                defaults.low_activity_ratio_per_sec,
            )
        ),
        low_activity_floor_per_sec=float(
            section.get(
                "low_activity_floor_per_sec",
                defaults.low_activity_floor_per_sec,
            )
        ),
    )


def _load_hold_config(section: dict[str, Any]) -> HoldConfig:
    defaults = HoldConfig()
    return HoldConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        activity_window_ms=int(
            section.get("activity_window_ms", defaults.activity_window_ms)
        ),
        ratio_per_sec_enter=float(
            section.get("ratio_per_sec_enter", defaults.ratio_per_sec_enter)
        ),
        ratio_per_sec_exit=float(
            section.get("ratio_per_sec_exit", defaults.ratio_per_sec_exit)
        ),
        floor_per_sec=float(section.get("floor_per_sec", defaults.floor_per_sec)),
        edge_margin_ratio=float(
            section.get("edge_margin_ratio", defaults.edge_margin_ratio)
        ),
    )


def _load_output_smoothing_config(section: dict[str, Any]) -> OutputSmoothingConfig:
    defaults = OutputSmoothingConfig()
    return OutputSmoothingConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        activity_window_ms=int(
            section.get("activity_window_ms", defaults.activity_window_ms)
        ),
        tau_active_s=float(section.get("tau_active_s", defaults.tau_active_s)),
        tau_extreme_s=float(section.get("tau_extreme_s", defaults.tau_extreme_s)),
        tau_hold_s=float(section.get("tau_hold_s", defaults.tau_hold_s)),
        activity_low_ratio_per_sec=float(
            section.get(
                "activity_low_ratio_per_sec",
                defaults.activity_low_ratio_per_sec,
            )
        ),
        activity_high_ratio_per_sec=float(
            section.get(
                "activity_high_ratio_per_sec",
                defaults.activity_high_ratio_per_sec,
            )
        ),
        activity_floor_per_sec=float(
            section.get(
                "activity_floor_per_sec",
                defaults.activity_floor_per_sec,
            )
        ),
        edge_margin_ratio=float(
            section.get("edge_margin_ratio", defaults.edge_margin_ratio)
        ),
    )


def _load_raw_qc_config(section: dict[str, Any]) -> RawQCConfig:
    defaults = RawQCConfig()
    return RawQCConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        raw_saturation_lo=float(
            section.get("raw_saturation_lo", defaults.raw_saturation_lo)
        ),
        raw_saturation_hi=float(
            section.get("raw_saturation_hi", defaults.raw_saturation_hi)
        ),
        flatline_epsilon=float(
            section.get("flatline_epsilon", defaults.flatline_epsilon)
        ),
        flatline_duration_s=float(
            section.get("flatline_duration_s", defaults.flatline_duration_s)
        ),
        baseline_ema_tau_s=float(
            section.get("baseline_ema_tau_s", defaults.baseline_ema_tau_s)
        ),
        baseline_abs_dev_tau_s=float(
            section.get(
                "baseline_abs_dev_tau_s",
                defaults.baseline_abs_dev_tau_s,
            )
        ),
        baseline_shift_sigma=float(
            section.get("baseline_shift_sigma", defaults.baseline_shift_sigma)
        ),
        baseline_shift_floor=float(
            section.get("baseline_shift_floor", defaults.baseline_shift_floor)
        ),
        warmup_s=float(section.get("warmup_s", defaults.warmup_s)),
    )


def _load_extrema_config(section: dict[str, Any]) -> ExtremaConfig:
    defaults = ExtremaConfig()
    return ExtremaConfig(
        min_interval_ms=int(section.get("min_interval_ms", defaults.min_interval_ms)),
        prominence_ratio=float(section.get("prominence_ratio", defaults.prominence_ratio)),
    )


def _load_output_config(section: dict[str, Any]) -> OutputConfig:
    defaults = OutputConfig()
    return OutputConfig(root_dir=str(section.get("root_dir", defaults.root_dir)))


def _validate_config(config: AppConfig) -> None:
    if config.device.sampling_rate_hz <= 0:
        raise ValueError("device.sampling_rate_hz must be positive.")
    if config.device.chunk_size <= 0:
        raise ValueError("device.chunk_size must be positive.")
    if config.device.processed_sensor_column < 0:
        raise ValueError("device.processed_sensor_column must be non-negative.")
    if not config.device.channels:
        raise ValueError("device.channels must contain at least one channel.")
    if config.display.plot_window_length <= 0:
        raise ValueError("display.plot_window_length must be positive.")
    if config.lsl.constant_delay_s < 0.0:
        raise ValueError("lsl.constant_delay_s must be non-negative.")
    _validate_filter_section(
        config.device.sampling_rate_hz,
        config.filter.hp_cutoff_hz,
        config.filter.hp_order,
        config.filter.lp_cutoff_hz,
        config.filter.lp_order,
        prefix="filter",
        validate_high_pass=False,
    )
    _validate_filter_section(
        config.device.sampling_rate_hz,
        config.movement.hp_cutoff_hz,
        config.movement.hp_order,
        config.movement.lp_cutoff_hz,
        config.movement.lp_order,
        prefix="movement",
    )
    if config.calibration.duration_s <= 0.0:
        raise ValueError("calibration.duration_s must be positive.")
    if config.calibration.amplitude_floor <= 0.0:
        raise ValueError("calibration.amplitude_floor must be positive.")
    if config.calibration.padding_ratio < 0.0:
        raise ValueError("calibration.padding_ratio must be non-negative.")
    if config.adaptation.low_activity_window_ms <= 0:
        raise ValueError("adaptation.low_activity_window_ms must be positive.")
    if config.adaptation.low_activity_ratio_per_sec <= 0.0:
        raise ValueError("adaptation.low_activity_ratio_per_sec must be positive.")
    if config.adaptation.low_activity_floor_per_sec <= 0.0:
        raise ValueError("adaptation.low_activity_floor_per_sec must be positive.")
    if config.hold.activity_window_ms <= 0:
        raise ValueError("hold.activity_window_ms must be positive.")
    if config.hold.ratio_per_sec_enter <= 0.0:
        raise ValueError("hold.ratio_per_sec_enter must be positive.")
    if config.hold.ratio_per_sec_exit <= config.hold.ratio_per_sec_enter:
        raise ValueError("hold.ratio_per_sec_exit must exceed hold.ratio_per_sec_enter.")
    if config.hold.floor_per_sec <= 0.0:
        raise ValueError("hold.floor_per_sec must be positive.")
    if not (0.0 < config.hold.edge_margin_ratio < 0.5):
        raise ValueError("hold.edge_margin_ratio must be between 0 and 0.5.")
    if config.output_smoothing.activity_window_ms <= 0:
        raise ValueError("output_smoothing.activity_window_ms must be positive.")
    if config.output_smoothing.tau_active_s <= 0.0:
        raise ValueError("output_smoothing.tau_active_s must be positive.")
    if config.output_smoothing.tau_extreme_s <= 0.0:
        raise ValueError("output_smoothing.tau_extreme_s must be positive.")
    if config.output_smoothing.tau_hold_s <= 0.0:
        raise ValueError("output_smoothing.tau_hold_s must be positive.")
    if config.output_smoothing.tau_extreme_s < config.output_smoothing.tau_active_s:
        raise ValueError(
            "output_smoothing.tau_extreme_s must be >= output_smoothing.tau_active_s."
        )
    if config.output_smoothing.tau_hold_s < config.output_smoothing.tau_extreme_s:
        raise ValueError(
            "output_smoothing.tau_hold_s must be >= output_smoothing.tau_extreme_s."
        )
    if config.output_smoothing.activity_low_ratio_per_sec <= 0.0:
        raise ValueError("output_smoothing.activity_low_ratio_per_sec must be positive.")
    if (
        config.output_smoothing.activity_high_ratio_per_sec
        <= config.output_smoothing.activity_low_ratio_per_sec
    ):
        raise ValueError(
            "output_smoothing.activity_high_ratio_per_sec must exceed output_smoothing.activity_low_ratio_per_sec."
        )
    if config.output_smoothing.activity_floor_per_sec <= 0.0:
        raise ValueError("output_smoothing.activity_floor_per_sec must be positive.")
    if not (0.0 < config.output_smoothing.edge_margin_ratio < 0.5):
        raise ValueError("output_smoothing.edge_margin_ratio must be between 0 and 0.5.")
    if config.extrema.min_interval_ms <= 0:
        raise ValueError("extrema.min_interval_ms must be positive.")
    if config.extrema.prominence_ratio <= 0.0:
        raise ValueError("extrema.prominence_ratio must be positive.")
    if config.raw_qc.raw_saturation_lo >= config.raw_qc.raw_saturation_hi:
        raise ValueError("raw_qc saturation bounds must be ordered.")
    if config.movement.low_activity_window_ms <= 0:
        raise ValueError("movement.low_activity_window_ms must be positive.")
    if config.movement.low_activity_ratio_per_sec <= 0.0:
        raise ValueError("movement.low_activity_ratio_per_sec must be positive.")
    if config.movement.low_activity_floor_per_sec <= 0.0:
        raise ValueError("movement.low_activity_floor_per_sec must be positive.")
    if not (0.0 <= config.movement.low_activity_drift_scale <= 1.0):
        raise ValueError("movement.low_activity_drift_scale must be between 0 and 1.")


def validate_live_acquisition_config(config: AppConfig) -> None:
    """Validate settings required specifically for live device acquisition."""

    if not config.device.mac_address.strip():
        raise ValueError("device.mac_address must be set for live acquisition.")


def _validate_filter_section(
    sampling_rate_hz: int,
    hp_cutoff_hz: float,
    hp_order: int,
    lp_cutoff_hz: float,
    lp_order: int,
    *,
    prefix: str,
    validate_high_pass: bool = True,
) -> None:
    nyquist_hz = sampling_rate_hz / 2.0
    if validate_high_pass:
        if hp_cutoff_hz <= 0.0 or hp_cutoff_hz >= nyquist_hz:
            raise ValueError(f"{prefix}.hp_cutoff_hz must be between 0 and Nyquist.")
        if hp_order <= 0:
            raise ValueError(f"{prefix}.hp_order must be positive.")
    if lp_cutoff_hz <= 0.0 or lp_cutoff_hz >= nyquist_hz:
        raise ValueError(f"{prefix}.lp_cutoff_hz must be between 0 and Nyquist.")
    if lp_order <= 0:
        raise ValueError(f"{prefix}.lp_order must be positive.")
