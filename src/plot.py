import matplotlib.pyplot as plt
import numpy as np

def plot_breathing_channel(channel_data, time=None, live=False, ax=None, line=None, blit_manager=None):
	"""
	Plots a single channel of breathing belt data.
	Args:
		channel_data: Array-like, the breathing signal values.
		time: Array-like or None, time axis (optional).
		live: bool, if True, updates an existing plot for live visualization.
		ax: matplotlib axis, for live plotting.
		line: matplotlib line object, for live plotting.
		blit_manager: BlitManager object for fast live updates (optional).
	"""
	# Always plot only the last 200 points
	channel_data = channel_data[-200:]
	# Keep plotted values inside the visible normalization range.
	channel_data = np.clip(np.asarray(channel_data, dtype=float), 0.0, 1.0)
	if time is not None:
		time = time[-200:]
	if live and ax is not None and line is not None:
		if time is not None:
			line.set_xdata(time)
			ax.set_xlim(min(time) if len(time) > 0 else 0, max(time) if len(time) > 0 else 1)
		else:
			line.set_xdata(range(len(channel_data)))
			ax.set_xlim(0, max(len(channel_data)-1, 1))
		line.set_ydata(channel_data)
		ax.set_ylim(0, 1)
		# Use regular redraw for scrolling axes; blitting can leave stale pixels.
		plt.pause(0.001)  # Only process GUI events, do not call plt.show()
	else:
		plt.figure(figsize=(10, 4))
		if time is not None:
			plt.plot(time, channel_data, label='Breathing Signal')
			plt.xlabel('Time (s)')
		else:
			plt.plot(channel_data, label='Breathing Signal')
			plt.xlabel('Sample')
		plt.ylabel('Amplitude')
		plt.title('Breathing Belt Channel Visualization')
		plt.ylim(0, 1)
		plt.legend()
		plt.tight_layout()
		plt.show()

def setup_live_plot(title='Breathing Belt Channel Visualization'):
	plt.ion()
	fig, ax = plt.subplots(figsize=(10, 4))
	line, = ax.plot([], [], label='Breathing Signal')
	ax.set_xlabel('Sample')
	ax.set_ylabel('Amplitude')
	ax.set_ylim(0, 1)
	ax.set_title(title)
	ax.legend()
	plt.tight_layout()
	plt.show(block=False)  # Show the window immediately
	plt.pause(0.01)  # Give time for window to appear
	# Disabled for scrolling x-limits to avoid stale/ghost rendering artifacts.
	blit_manager = None
	return fig, ax, line, blit_manager


# --- BlitManager utility for fast live plotting ---
class BlitManager:
	def __init__(self, canvas, animated_artists):
		self.canvas = canvas
		self.animated_artists = animated_artists
		for a in self.animated_artists:
			a.set_animated(True)
		self.background = None
		self._cid = self.canvas.mpl_connect("draw_event", self.on_draw)
		self.on_draw(None)

	def on_draw(self, event):
		self.background = self.canvas.figure.canvas.copy_from_bbox(self.canvas.figure.bbox)
		for a in self.animated_artists:
			self.canvas.figure.draw_artist(a)
		self.canvas.flush_events()

	def update(self):
		if self.background is not None:
			self.canvas.figure.canvas.restore_region(self.background)
			for a in self.animated_artists:
				self.canvas.figure.draw_artist(a)
			self.canvas.figure.canvas.blit(self.canvas.figure.bbox)
			self.canvas.flush_events()
