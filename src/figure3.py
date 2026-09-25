#%%
import matplotlib.pyplot as plt
import numpy as np

from retrieve_model import retrieve_model

def plot_panel_a(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	# x = np.linspace(0, 10, 1000)
	ax.text(.5,.5,'SHD', horizontalalignment='center')
	ax.set_title("Task")
	ax.set_xticks([])
	ax.set_yticks([])
	return ax


def plot_panel_b(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	x = np.linspace(-10, 10, 1000)
	ax.plot(x, 8*np.maximum(x, 0))
	ax.set_title("Performance over initial E:I Ratio")
	ax.set_xlabel("E:I Ratio")
	ax.set_ylabel("Accuracy")
	ax.set_ylim(-10,110)
	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()
	# ax.set_ylabel("cos(x)")
	return ax



def plot_panel_c(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	x = np.linspace(0, 8, 9)
	plt.bar(x, (4*x) + 50)
	ax.set_title('Performance vs. Point Neuron')
	ax.set_xlabel('Number of Branches [1]')
	ax.set_ylim(-10, 110)
	# ax.set_xticks(np.arange(0,9))

	return ax


def plot_panel_d(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	x = np.linspace(0, 10, 1000)
	ax.plot(np.exp(-.3*x), label='Dendritic Neuron')
	ax.plot(np.exp(-3.*x), label='Point Neuron')
	ax.set_title('Long-Term Spike Correlations')
	ax.set_yticks([])
	ax.set_xlabel('Time [ms]')
	ax.legend(loc='upper right')

	return ax



def plot_panel_e(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	# x = np.linspace(0, 10, 1000)
	ax.imshow(np.random.uniform(0,1, (32, 1000)) > .5)
	ax.set_title('Pre-Training Spiking Activity')
	ax.set_xlabel('Time [ms]')

	return ax


def plot_panel_f(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	x = np.linspace(0, 10, 1000)
	ax.imshow((np.sin(x[:,None] * 19 * x[:,None]) > 0).T)
	ax.set_title('Post-Training Spiking Activity')
	ax.set_yticks([])
	ax.set_xlabel('Time [ms]')

	return ax


def plot_panel_g(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	x = np.linspace(0, 10, 1000)
	noise = np.random.standard_normal(x.shape) * .025
	ax.plot(x, np.exp(-x) + noise)
	ax.plot(x, 1.5*np.exp(-x) + noise)
	ax.plot(x, 1.5*np.exp(-.1*x) + noise)
	ax.set_title('Accuracy Over Time with Noise')
	ax.set_xlabel('Time [ms]')
	ax.set_xticks(np.arange(0,10, 1))
	ax.set_xticklabels(np.arange(0,10, 1)*100)

	return ax


fig = plt.figure(figsize=(10, 7))

gs = fig.add_gridspec(
	4, 2,
	width_ratios=[1, 1],
	height_ratios=[1, 1, 1, 1],
	hspace=1,
	wspace=0.05,
)

plot_panel_a(fig.add_subplot(gs[0, 0]))
plot_panel_b(fig.add_subplot(gs[0, 1]))
plot_panel_c(fig.add_subplot(gs[1, 0]))
plot_panel_d(fig.add_subplot(gs[1, 1]))
plot_panel_e(fig.add_subplot(gs[2, 0]))
plot_panel_f(fig.add_subplot(gs[2, 1]))
plot_panel_g(fig.add_subplot(gs[3, :]))
plot_panel_g(fig.add_subplot(gs[3, :]))

plt.show()
# %%
