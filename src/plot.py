import matplotlib.pyplot as plt

def plot_breathing_channel(channel_data, time=None, live=False, ax=None, line=None):
	"""
	Plots a single channel of breathing belt data.
	Args:
		channel_data: Array-like, the breathing signal values.
		time: Array-like or None, time axis (optional).
		live: bool, if True, updates an existing plot for live visualization.
		ax: matplotlib axis, for live plotting.
		line: matplotlib line object, for live plotting.
	"""
	if live and ax is not None and line is not None:
		if time is not None:
			line.set_xdata(time)
		else:
			line.set_xdata(range(len(channel_data)))
		line.set_ydata(channel_data)
		ax.relim()
		ax.autoscale_view()
		plt.draw()
		plt.pause(0.01)
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
		plt.legend()
		plt.tight_layout()
		plt.show()

def setup_live_plot(title='Breathing Belt Channel Visualization'):
	plt.ion()
	fig, ax = plt.subplots(figsize=(10, 4))
	line, = ax.plot([], [], label='Breathing Signal')
	ax.set_xlabel('Sample')
	ax.set_ylabel('Amplitude')
	ax.set_title(title)
	ax.legend()
	plt.tight_layout()
	return fig, ax, line
