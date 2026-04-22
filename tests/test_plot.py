"""Tests for live raw and normalized plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.plot import setup_live_plots, update_live_plots


def test_setup_live_plots_creates_side_by_side_dashboard() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, blit_manager = setup_live_plots()

    try:
        assert len(fig.axes) == 2
        assert raw_ax.get_title() == "Raw Sensor Signal"
        assert normalized_ax.get_title() == "Breath Level (0-1)"
        assert raw_line in raw_ax.lines
        assert normalized_line in normalized_ax.lines
        assert blit_manager is None
    finally:
        plt.close(fig)


def test_update_live_plots_keeps_raw_values_and_autoscales_axis() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots()

    try:
        raw_values = np.array([100.0, 200.0, 300.0], dtype=float)
        raw_time = np.array([10.0, 11.0, 12.0], dtype=float)
        normalized_values = np.array([0.2, 0.4, 0.6], dtype=float)
        normalized_time = np.array([10.0, 11.0, 12.0], dtype=float)

        update_live_plots(
            raw_values,
            raw_time,
            normalized_values,
            normalized_time,
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
        )

        assert np.allclose(raw_line.get_xdata(), raw_time)
        assert np.allclose(raw_line.get_ydata(), raw_values)
        raw_y_limits = raw_ax.get_ylim()
        assert raw_y_limits[0] < float(np.min(raw_values))
        assert raw_y_limits[1] > float(np.max(raw_values))
    finally:
        plt.close(fig)


def test_update_live_plots_keeps_full_history_and_shared_x_axis() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots()

    try:
        raw_values = np.linspace(100.0, 399.0, 300, dtype=float)
        raw_time = np.arange(300, dtype=float)
        normalized_values = np.linspace(0.2, 0.8, 150, dtype=float)
        normalized_time = np.arange(150, 300, dtype=float)

        update_live_plots(
            raw_values,
            raw_time,
            normalized_values,
            normalized_time,
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
        )

        assert len(raw_line.get_xdata()) == 300
        assert np.allclose(raw_line.get_xdata(), raw_time)
        assert np.allclose(normalized_line.get_xdata(), normalized_time)
        assert raw_ax.get_xlim() == (0.0, 299.0)
        assert normalized_ax.get_xlim() == (0.0, 299.0)
    finally:
        plt.close(fig)


def test_update_live_plots_adds_peak_and_trough_markers_to_raw_axis() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots()

    try:
        update_live_plots(
            raw_channel_data=np.array([10.0, 20.0, 30.0], dtype=float),
            raw_time=np.array([0.0, 1.0, 2.0], dtype=float),
            normalized_channel_data=np.array([0.2, 0.4, 0.6], dtype=float),
            normalized_time=np.array([0.0, 1.0, 2.0], dtype=float),
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
            peak_times=np.array([1.0], dtype=float),
            peak_values=np.array([20.0], dtype=float),
            trough_times=np.array([2.0], dtype=float),
            trough_values=np.array([30.0], dtype=float),
        )

        assert len(raw_ax.collections) == 2
    finally:
        plt.close(fig)


def test_update_live_plots_preserves_unrelated_raw_axis_collections() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots()

    try:
        extra_artist = raw_ax.scatter(
            np.array([0.0], dtype=float),
            np.array([5.0], dtype=float),
            color="tab:green",
            s=25,
            label="_nolegend_",
        )

        update_live_plots(
            raw_channel_data=np.array([10.0, 20.0, 30.0], dtype=float),
            raw_time=np.array([0.0, 1.0, 2.0], dtype=float),
            normalized_channel_data=np.array([0.2, 0.4, 0.6], dtype=float),
            normalized_time=np.array([0.0, 1.0, 2.0], dtype=float),
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
            peak_times=np.array([1.0], dtype=float),
            peak_values=np.array([20.0], dtype=float),
            trough_times=np.array([2.0], dtype=float),
            trough_values=np.array([30.0], dtype=float),
        )
        update_live_plots(
            raw_channel_data=np.array([10.0, 20.0, 30.0], dtype=float),
            raw_time=np.array([0.0, 1.0, 2.0], dtype=float),
            normalized_channel_data=np.array([0.2, 0.4, 0.6], dtype=float),
            normalized_time=np.array([0.0, 1.0, 2.0], dtype=float),
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
            peak_times=np.array([0.0], dtype=float),
            peak_values=np.array([10.0], dtype=float),
            trough_times=np.array([1.0], dtype=float),
            trough_values=np.array([20.0], dtype=float),
        )

        assert extra_artist in raw_ax.collections
        assert len(raw_ax.collections) == 3
    finally:
        plt.close(fig)


def test_update_live_plots_clips_normalized_trace_and_keeps_fixed_range() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots()

    try:
        update_live_plots(
            raw_channel_data=np.array([10.0, 20.0], dtype=float),
            raw_time=np.array([0.0, 1.0], dtype=float),
            normalized_channel_data=np.array([-0.25, 0.5, 1.25], dtype=float),
            normalized_time=np.array([0.0, 1.0, 2.0], dtype=float),
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
        )

        assert np.allclose(normalized_line.get_ydata(), np.array([0.0, 0.5, 1.0], dtype=float))
        assert normalized_ax.get_ylim() == (0.0, 1.0)
    finally:
        plt.close(fig)


def test_update_live_plots_uses_non_degenerate_limits_for_flat_raw_window() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots()

    try:
        flat_raw = np.array([512.0, 512.0, 512.0], dtype=float)
        update_live_plots(
            raw_channel_data=flat_raw,
            raw_time=np.array([5.0, 6.0, 7.0], dtype=float),
            normalized_channel_data=np.array([], dtype=float),
            normalized_time=np.array([], dtype=float),
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
        )

        raw_y_limits = raw_ax.get_ylim()
        assert raw_y_limits[1] > raw_y_limits[0]
        assert np.isclose(0.5 * (raw_y_limits[0] + raw_y_limits[1]), 512.0)
        assert (raw_y_limits[1] - raw_y_limits[0]) >= 1.0
    finally:
        plt.close(fig)


def test_update_live_plots_supports_autoscaled_movement_trace() -> None:
    fig, raw_ax, raw_line, normalized_ax, normalized_line, _ = setup_live_plots(
        normalized_title="Movement Proxy (Centered)",
        normalized_label="Movement Proxy",
    )

    try:
        movement_values = np.array([-5.0, 0.0, 8.0], dtype=float)
        update_live_plots(
            raw_channel_data=np.array([10.0, 20.0, 30.0], dtype=float),
            raw_time=np.array([0.0, 1.0, 2.0], dtype=float),
            normalized_channel_data=movement_values,
            normalized_time=np.array([0.0, 1.0, 2.0], dtype=float),
            raw_ax=raw_ax,
            raw_line=raw_line,
            normalized_ax=normalized_ax,
            normalized_line=normalized_line,
            normalized_clip_range=None,
            normalized_fixed_ylim=None,
            normalized_autoscale_y=True,
        )

        assert normalized_ax.get_title() == "Movement Proxy (Centered)"
        assert np.allclose(normalized_line.get_ydata(), movement_values)
        normalized_y_limits = normalized_ax.get_ylim()
        assert normalized_y_limits[0] < -5.0
        assert normalized_y_limits[1] > 8.0
    finally:
        plt.close(fig)
