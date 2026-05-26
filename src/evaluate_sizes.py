#%%
import os

from ray import tune
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


from config.ray.pbt import CONFIG
from ray_training import build_model, build_test_loader
from utils.settings import DEVICE
from utils.diagnosis import count_trainable_parameters
from retrieve_history import retrieve_history

CONFIG['dt'] = 1e-1

# %%
directory = "/Users/cankayser/Downloads/results/"


def retrieve_model_metadata(directory):
	t, acc = retrieve_history(directory)
	e = tune.ExperimentAnalysis(directory)

	best_trial = e.get_best_trial(metric='evaluation', mode='min', scope='all')

	model = build_model(best_trial.config)
	params = count_trainable_parameters(model)

	conf = best_trial.config

	return np.nanmax(acc), conf['n_hidden'], conf['n_dendrites'], params

def heatmap(data, row_labels, col_labels, ax=None,
			cbar_kw=None, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
		A 2D numpy array of shape (M, N).
	row_labels
		A list or array of length M with the labels for the rows.
	col_labels
		A list or array of length N with the labels for the columns.
	ax
		A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
		not provided, use current Axes or create a new one.  Optional.
	cbar_kw
		A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
		The label for the colorbar.  Optional.
	**kwargs
		All other arguments are forwarded to `imshow`.
	"""

	if ax is None:
		ax = plt.gca()

	if cbar_kw is None:
		cbar_kw = {}

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel)

	# Show all ticks and label them with the respective list entries.
	ax.set_xticks(range(data.shape[1]), labels=col_labels)
	ax.set_yticks(range(data.shape[0]), labels=row_labels)


	# Turn spines off and create white grid.
	ax.spines[:].set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	# ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)

	return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
					 textcolors=("black", "white"),
					 threshold=None, **textkw):
	"""
	A function to annotate a heatmap.

	Parameters
	----------
	im
		The AxesImage to be labeled.
	data
		Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
		The format of the annotations inside the heatmap.  This should either
		use the string format method, e.g. "$ {x:.2f}", or be a
		`matplotlib.ticker.Formatter`.  Optional.
	textcolors
		A pair of colors.  The first is used for values below a threshold,
		the second for those above.  Optional.
	threshold
		Value in data units according to which the colors from textcolors are
		applied.  If None (the default) uses the middle of the colormap as
		separation.  Optional.
	**kwargs
		All other arguments are forwarded to each call to `text` used to create
		the text labels.
	"""

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = mpl.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)

	return texts

# %%
# %%
if __name__ == '__main__':
	acc_list = []
	params_list = []
	neuron_list = []
	dendrite_list = []
	for model_directory in os.listdir(directory):
		try:
			a, n, d, p = retrieve_model_metadata(directory + model_directory)
		except NotADirectoryError:
			continue

		acc_list.append(a)
		params_list.append(p)
		neuron_list.append(n)
		dendrite_list.append(d)

	all_n = np.array(list(set(neuron_list)))
	all_d = np.array(list(set(dendrite_list)))

	all_a = np.full((len(all_n), len(all_d)), np.nan)
	all_p = np.full((len(all_n), len(all_d)), np.nan)

	for a, n, d, p in zip(acc_list, neuron_list, dendrite_list, params_list):
		i  = np.where(all_n == n)[0][0]
		j  = np.where(all_d == d)[0][0]
		all_a[i,j] = a
		all_p[i,j] = p

	with plt.style.context('save'):
		fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, layout='compressed')
		plt.subplot(211)
		im, cbar = heatmap(all_a, all_n, all_d, cmap='magma_r', cbarlabel=r'Accuracy $[\%]$')
		cbar.set_ticks([])
		# annotate_heatmap(im, valfmt=r"{x:1.0f} $\%$", textcolors=('w','k'))
		annotate_heatmap(im, valfmt=r"{x:1.0f} $\%$")
		plt.ylabel(r'Number of Neurons [1]')

		plt.subplot(212)
		im, cbar = heatmap(all_p / 1e3, all_n, all_d, cmap='magma_r', cbarlabel=r'Number of Parameters $[10^3]$')
		cbar.set_ticks([])
		# annotate_heatmap(im, valfmt=r"{x:1.0f}", textcolors=('w','k'))
		annotate_heatmap(im, valfmt=r"{x:1.0f}")
		plt.ylabel(r'Number of Neurons [1]')
		plt.xlabel(r'Number of Dendrites [1]')
		plt.savefig('../thesis/figures/evaluate_sizes')
		plt.show()




	# %%
