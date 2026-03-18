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
        assert normalized_ax.get_title() == "Normalized Breathing Signal (0-1 range)"
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
