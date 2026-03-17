"""Plotting utilities for live and offline respiration-belt visualization."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.lines import Line2D
from matplotlib.figure import Figure


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

    channel_data = channel_data[-200:]
    # Clipping is applied only to the displayed trace. The underlying signal
    # used for calibration and streaming is unchanged.
    channel_data = np.clip(np.asarray(channel_data, dtype=float), 0.0, 1.0)
    if time is not None:
        time = time[-200:]

    if live and ax is not None and line is not None:
        if time is not None:
            line.set_xdata(time)
            ax.set_xlim(min(time) if len(time) > 0 else 0, max(time) if len(time) > 0 else 1)
        else:
            line.set_xdata(range(len(channel_data)))
            ax.set_xlim(0, max(len(channel_data) - 1, 1))

        line.set_ydata(channel_data)
        ax.set_ylim(0, 1)
        # Use a full redraw for scrolling axes; blitting leaves stale pixels in
        # this usage mode more often than it improves performance.
        plt.pause(0.001)
        return

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
    plt.show(block=False)
    plt.pause(0.01)

    # Disabled for scrolling x-limits to avoid stale or ghost rendering artifacts.
    blit_manager = None
    return fig, ax, line, blit_manager


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
