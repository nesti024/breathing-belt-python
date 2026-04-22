"""Plotting utilities for live and offline respiration-belt visualization."""

from __future__ import annotations

from typing import Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


_RAW_Y_PADDING_RATIO = 0.05
_RAW_MIN_Y_SPAN = 1.0


def plot_breathing_channel(
    channel_data: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray | None = None,
    live: bool = False,
    ax: Axes | None = None,
    line: Line2D | None = None,
    blit_manager: "BlitManager | None" = None,
) -> None:
    """Plot a respiration trace or update an existing live plot.

    Parameters
    ----------
    channel_data:
        Array-like respiration signal values.
    time:
        Optional time or sample axis aligned with ``channel_data``.
    live:
        If ``True``, update an existing Matplotlib line in place.
    ax, line, blit_manager:
        Existing Matplotlib objects for live visualization.
    """

    del blit_manager  # Reserved for future optimization; unused in scrolling mode.

    if live and ax is not None and line is not None:
        _update_live_line(
            channel_data=channel_data,
            time=time,
            ax=ax,
            line=line,
            clip_range=(0.0, 1.0),
            fixed_ylim=(0.0, 1.0),
        )
        _refresh_live_canvas()
        return

    # Clipping is applied only to the displayed trace. The underlying signal
    # used for calibration and streaming is unchanged.
    channel_data = np.clip(np.asarray(channel_data, dtype=float), 0.0, 1.0)

    plt.figure(figsize=(10, 4))
    if time is not None:
        plt.plot(time, channel_data, label="Breathing Signal")
        plt.xlabel("Time (s)")
    else:
        plt.plot(channel_data, label="Breathing Signal")
        plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Breathing Belt Channel Visualization")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def setup_live_plot(
    title: str = "Breathing Belt Channel Visualization",
) -> tuple[Figure, Axes, Line2D, None]:
    """Create and show an interactive plot for the normalized respiration trace."""

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], label="Breathing Signal")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _show_live_figure()

    # Disabled for scrolling x-limits to avoid stale or ghost rendering artifacts.
    blit_manager = None
    return fig, ax, line, blit_manager


def setup_live_plots(
    *,
    raw_title: str = "Raw Sensor Signal",
    normalized_title: str = "Breath Level (0-1)",
    normalized_label: str = "Breath Level",
    normalized_ylabel: str = "Amplitude",
) -> tuple[Figure, Axes, Line2D, Axes, Line2D, None]:
    """Create and show a two-panel live dashboard for raw and normalized traces."""

    plt.ion()
    fig, (raw_ax, normalized_ax) = plt.subplots(1, 2, figsize=(14, 4))
    raw_line, = raw_ax.plot([], [], label="Raw Sensor Signal")
    normalized_line, = normalized_ax.plot([], [], label=normalized_label)

    raw_ax.set_xlabel("Sample")
    raw_ax.set_ylabel("Raw Amplitude")
    raw_ax.set_xlim(0, 1)
    raw_ax.set_ylim(0, 1)
    raw_ax.set_title(raw_title)
    raw_ax.legend()

    normalized_ax.set_xlabel("Sample")
    normalized_ax.set_ylabel(normalized_ylabel)
    normalized_ax.set_xlim(0, 1)
    normalized_ax.set_ylim(0, 1)
    normalized_ax.set_title(normalized_title)
    normalized_ax.legend()

    plt.tight_layout()
    _show_live_figure()

    # Disabled for scrolling x-limits to avoid stale or ghost rendering artifacts.
    blit_manager = None
    return fig, raw_ax, raw_line, normalized_ax, normalized_line, blit_manager


def update_live_plots(
    raw_channel_data: Sequence[float] | np.ndarray,
    raw_time: Sequence[float] | np.ndarray | None,
    normalized_channel_data: Sequence[float] | np.ndarray,
    normalized_time: Sequence[float] | np.ndarray | None,
    *,
    raw_ax: Axes,
    raw_line: Line2D,
    normalized_ax: Axes,
    normalized_line: Line2D,
    peak_times: Sequence[float] | np.ndarray | None = None,
    peak_values: Sequence[float] | np.ndarray | None = None,
    trough_times: Sequence[float] | np.ndarray | None = None,
    trough_values: Sequence[float] | np.ndarray | None = None,
    normalized_clip_range: tuple[float, float] | None = (0.0, 1.0),
    normalized_fixed_ylim: tuple[float, float] | None = (0.0, 1.0),
    normalized_autoscale_y: bool = False,
    blit_manager: "BlitManager | None" = None,
) -> None:
    """Update the live raw and normalized plots in one redraw pass."""

    del blit_manager  # Reserved for future optimization; unused in scrolling mode.
    shared_x_limits = _compute_shared_x_limits(
        _coerce_x_values(raw_channel_data, raw_time),
        _coerce_x_values(normalized_channel_data, normalized_time),
    )

    _update_live_line(
        channel_data=raw_channel_data,
        time=raw_time,
        ax=raw_ax,
        line=raw_line,
        x_limits=shared_x_limits,
        autoscale_y=True,
        min_y_span=_RAW_MIN_Y_SPAN,
        y_padding_ratio=_RAW_Y_PADDING_RATIO,
    )
    _update_live_line(
        channel_data=normalized_channel_data,
        time=normalized_time,
        ax=normalized_ax,
        line=normalized_line,
        x_limits=shared_x_limits,
        clip_range=normalized_clip_range,
        fixed_ylim=normalized_fixed_ylim,
        autoscale_y=normalized_autoscale_y,
        min_y_span=_RAW_MIN_Y_SPAN,
        y_padding_ratio=_RAW_Y_PADDING_RATIO,
    )
    _update_raw_event_markers(
        raw_ax,
        peak_times=peak_times,
        peak_values=peak_values,
        trough_times=trough_times,
        trough_values=trough_values,
    )
    _refresh_live_canvas()


def _update_live_line(
    *,
    channel_data: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray | None,
    ax: Axes,
    line: Line2D,
    clip_range: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
    autoscale_y: bool = False,
    min_y_span: float = _RAW_MIN_Y_SPAN,
    y_padding_ratio: float = _RAW_Y_PADDING_RATIO,
    x_limits: tuple[float, float] | None = None,
) -> None:
    values = np.asarray(channel_data, dtype=float)
    if clip_range is not None:
        values = np.clip(values, clip_range[0], clip_range[1])

    x_values = _coerce_x_values(values, time)

    line.set_xdata(x_values)
    line.set_ydata(values)
    if x_limits is None:
        ax.set_xlim(*_compute_x_limits(x_values))
    else:
        ax.set_xlim(*x_limits)

    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif autoscale_y:
        ax.set_ylim(
            *_compute_padded_y_limits(
                values,
                min_y_span=min_y_span,
                padding_ratio=y_padding_ratio,
            )
        )


def _compute_x_limits(x_values: np.ndarray) -> tuple[float, float]:
    if x_values.size == 0:
        return 0.0, 1.0

    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    if x_min == x_max:
        x_max = x_min + 1.0
    return x_min, x_max


def _coerce_x_values(
    channel_data: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(channel_data, dtype=float)
    if time is None:
        return np.arange(values.size, dtype=float)

    x_values = np.asarray(time, dtype=float)
    if x_values.shape != values.shape:
        raise ValueError("time and channel_data must have matching shapes for live updates.")
    return x_values


def _compute_shared_x_limits(*x_series: np.ndarray) -> tuple[float, float]:
    non_empty_series = [series for series in x_series if series.size > 0]
    if not non_empty_series:
        return 0.0, 1.0

    x_min = min(float(np.min(series)) for series in non_empty_series)
    x_max = max(float(np.max(series)) for series in non_empty_series)
    if x_min == x_max:
        x_max = x_min + 1.0
    return x_min, x_max


def _compute_padded_y_limits(
    values: np.ndarray,
    *,
    min_y_span: float,
    padding_ratio: float,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = y_max - y_min
    if span < min_y_span:
        center = 0.5 * (y_min + y_max)
        half_span = 0.5 * min_y_span
        return center - half_span, center + half_span

    padding = span * padding_ratio
    return y_min - padding, y_max + padding


def _refresh_live_canvas() -> None:
    # Use a full redraw for scrolling axes; blitting leaves stale pixels in
    # this usage mode more often than it improves performance.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        plt.pause(0.001)


def _update_raw_event_markers(
    ax: Axes,
    *,
    peak_times: Sequence[float] | np.ndarray | None,
    peak_values: Sequence[float] | np.ndarray | None,
    trough_times: Sequence[float] | np.ndarray | None,
    trough_values: Sequence[float] | np.ndarray | None,
) -> None:
    _replace_raw_event_marker(
        ax,
        artist_attr_name="_peak_marker_artist",
        times=peak_times,
        values=peak_values,
        marker="^",
        color="tab:red",
    )
    _replace_raw_event_marker(
        ax,
        artist_attr_name="_trough_marker_artist",
        times=trough_times,
        values=trough_values,
        marker="v",
        color="tab:blue",
    )


def _replace_raw_event_marker(
    ax: Axes,
    *,
    artist_attr_name: str,
    times: Sequence[float] | np.ndarray | None,
    values: Sequence[float] | np.ndarray | None,
    marker: str,
    color: str,
) -> None:
    existing_artist = getattr(ax, artist_attr_name, None)
    if existing_artist is not None:
        existing_artist.remove()
        setattr(ax, artist_attr_name, None)

    if times is None or values is None:
        return

    marker_x = np.asarray(times, dtype=float)
    marker_y = np.asarray(values, dtype=float)
    if marker_x.size == 0 or marker_y.size == 0:
        return

    artist = ax.scatter(
        marker_x,
        marker_y,
        marker=marker,
        color=color,
        s=36,
        zorder=3,
        label="_nolegend_",
    )
    setattr(ax, artist_attr_name, artist)


def _show_live_figure() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        plt.show(block=False)
        plt.pause(0.01)


class BlitManager:
    """Minimal blitting helper retained for non-scrolling plot use cases."""

    def __init__(
        self,
        canvas: FigureCanvasBase,
        animated_artists: Sequence,
    ) -> None:
        self.canvas = canvas
        self.animated_artists = animated_artists
        for artist in self.animated_artists:
            artist.set_animated(True)
        self.background = None
        self._cid = self.canvas.mpl_connect("draw_event", self.on_draw)
        self.on_draw(None)

    def on_draw(self, event) -> None:
        del event
        self.background = self.canvas.figure.canvas.copy_from_bbox(self.canvas.figure.bbox)
        for artist in self.animated_artists:
            self.canvas.figure.draw_artist(artist)
        self.canvas.flush_events()

    def update(self) -> None:
        if self.background is None:
            return

        self.canvas.figure.canvas.restore_region(self.background)
        for artist in self.animated_artists:
            self.canvas.figure.draw_artist(artist)
        self.canvas.figure.canvas.blit(self.canvas.figure.bbox)
        self.canvas.flush_events()
