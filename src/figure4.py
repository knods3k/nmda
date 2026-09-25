#%%
import os


import matplotlib.pyplot as plt
import numpy as np
import torch
from ray import tune

from ray_training import build_model
from utils.settings import DEVICE
from utils.diagnosis import get_n_dendrites, get_n_neurons, count_trainable_parameters
from retrieve_model import retrieve_model_with_config, retrieve_model
from datasets.shd_classification import build_test_loader
from model import DendriticSNN_coupled, LIF_SNN

DIR = '/Users/cankayser/Downloads/ray_results'
best_model_dir = '/Users/cankayser/Downloads/ray_results/63360'
N_test = 128

#%%

test_loader = build_test_loader(1)
test_x, test_y = next(iter(test_loader))
test_x = test_x.to(DEVICE)
test_y = test_y.to(DEVICE)

best_model, config = retrieve_model_with_config(best_model_dir)
best_model = best_model.to(DEVICE)
states = best_model.test(test_x.to(DEVICE))

model_pretrain = build_model(config).to(DEVICE)
states_pretrain = model_pretrain.test(test_x.to(DEVICE))


# best_model_point = retrieve_model(os.path.join(DIR, '63699')).to(DEVICE)

#%%

def plot_panel_a(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	# x = np.linspace(0, 10, 1000)
	# ax.plot(x, np.sin(x))
	ax.set_title("Evaluation Protocol")
	ax.set_xticks([])
	ax.set_yticks([])
	return ax

def plot_panel_b(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	test_loader = build_test_loader(N_test)
	test_x, test_y = next(iter(test_loader))
	test_x = test_x.to(DEVICE)
	test_y = test_y.to(DEVICE)

	accuracies_dend = []
	accuracies_point = []
	n_params_dend = []
	n_params_point = []
	for d in os.listdir(DIR):
		if d == '.DS_Store':
			continue
		path = os.path.join(DIR, d)
		model = retrieve_model(path).to(DEVICE)
		y_pred = model(test_x).mean(1).argmax(-1)
		acc = (y_pred == test_y).float().mean().cpu().numpy() * 100
		if isinstance(model, DendriticSNN_coupled):
			accuracies_dend.append(acc)
			n_params_dend.append(count_trainable_parameters(model))
		elif isinstance(model, LIF_SNN):
			accuracies_point.append(acc)
			n_params_point.append(count_trainable_parameters(model))


	# x = np.linspace(0, 10, 1000)
	ax.scatter(n_params_dend,accuracies_dend, label='DendriticSNN')
	ax.scatter(n_params_point,accuracies_point, label='LIF_SNN')
	ax.set_title("Performance vs. Point Neuron")
	ax.set_xscale('log')
	ax.set_ylim(0,100)
	ax.legend()
	# ax.set_xlabel("x")
	# ax.set_ylabel("cos(x)")
	return ax


def plot_panel_c(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')


	state = list(states_pretrain[0].values())[0][0].T.cpu().numpy()

	ax.imshow(
		state,
		cmap='berlin',
		# vmin=-1,
		# vmax=1,
		)
	ax.set_title('Pre-Training Dendritic Activity')
	ax.set_yticks([])
	ax.set_xlabel('Time [ms]')

	ticks = np.arange(0,state.shape[1],10)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks*10)

	ax.set_xticks([])
	ax.set_xticklabels([])


	return ax

def plot_panel_d(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')


	state =	list(states[0].values())[0][0].T.cpu().numpy()

	ax.imshow(
		state,
		cmap='berlin',
		# vmin=-1,
		# vmax=1
		)
	ax.set_title('Post-Training Dendritic Activity')
	ax.set_yticks([])
	ax.set_xlabel('Time [ms]')

	ticks = np.arange(0,state.shape[1],10)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks*10)

	ax.set_xticks([])
	ax.set_xticklabels([])

	return ax


def plot_panel_e(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')


	state = list(states_pretrain[1].values())[0][0].T.cpu().numpy()

	ax.imshow(
		state,
		cmap='gray')
	ax.set_title('Pre-Training Spiking Activity')
	ax.set_yticks([])
	ax.set_xlabel('Time [ms]')

	ticks = np.arange(0,state.shape[1],10)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks*10)


	return ax


def plot_panel_f(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	state = list(states[1].values())[0][0].T.cpu().numpy()

	ax.imshow(
		state,
		cmap='gray'
		)
	ax.set_title('Post-Training Spiking Activity')
	ax.set_yticks([])
	ax.set_xlabel('Time [ms]')

	ticks = np.arange(0,state.shape[1],10)
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticks*10)

	return ax



def plot_panel_g(ax=None):
	if ax is None:
		_, ax = plt.subplots(layout='constrained')

	MAX_DELAY = 367
	NOISE_RATE = 1e-3

	DIRs = [
		'/Users/cankayser/Downloads/ray_results/63700',
		'/Users/cankayser/Downloads/ray_results/64024',
		'/Users/cankayser/Downloads/ray_results/63360',
	]

	loader = build_test_loader(N_test, delay_steps=MAX_DELAY)
	test_x, test_y = next(iter(loader))
	test_x = test_x.to(DEVICE)
	test_y = test_y.to(DEVICE)


	noise = torch.tensor(
		np.random.poisson(NOISE_RATE, test_x.shape[1:])
	).to(DEVICE)
	noise[:120,:] = 0


	for d in DIRs:
		path = os.path.join(DIR, d)
		model = retrieve_model(path).to(DEVICE)
		y_pred = model(test_x + noise)[:,-MAX_DELAY:,:].argmax(-1).T

		accuracies = np.zeros(MAX_DELAY)
		for i, delay in enumerate(range(MAX_DELAY)):
			p = (y_pred == test_y)[delay-MAX_DELAY,:].float().mean()
			accuracies[i] = p*100

		label = f"{type(model).__name__}_{get_n_neurons(model)}_{get_n_dendrites(model)}"
		ax.plot(accuracies, label=label)

	ax.set_title(f'Accuracy Over Time with {NOISE_RATE*100*10}% Noise')
	ax.set_xlabel('Time [ms]')
	ax.set_ylabel('Accuracy [%]')
	ax.set_ylim(0, 100)
	ax.set_xticklabels(np.arange(0,100, 15))
	ax.legend()

	return ax


fig = plt.figure(figsize=(12, 9))

gs = fig.add_gridspec(
	4, 2,
	width_ratios=[1, 1],
	height_ratios=[1, 1, 1, 1],
	hspace=.8,
	wspace=0.1,
)

plot_panel_a(fig.add_subplot(gs[0, 0]))
plot_panel_b(fig.add_subplot(gs[0, 1]))
plot_panel_c(fig.add_subplot(gs[1, 0]))
plot_panel_d(fig.add_subplot(gs[1, 1]))
plot_panel_e(fig.add_subplot(gs[2, 0]))
plot_panel_f(fig.add_subplot(gs[2, 1]))
plot_panel_g(fig.add_subplot(gs[3, :]))

plt.show()

# %%

# %%
