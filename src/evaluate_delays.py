#%%
from os import listdir

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from ray_training import build_test_loader, build_evaluation_function
from retrieve_model import retrieve_model
from utils.settings import DEVICE
from utils.diagnosis import get_n_neurons, get_n_dendrites, count_trainable_parameters

N = 2264
MAX_DELAY = 367

# N = N//20
# MAX_DELAY = MAX_DELAY//5

cmap = plt.get_cmap('Set1',lut=4)

def evaluate_delays(model, test_x, test_y, max_delay):
	delays = np.arange(0,max_delay)
	accuracies_mean = np.zeros_like(delays, dtype=float)
	accuracies_std = np.zeros_like(delays, dtype=float)
	with torch.no_grad():
		model.eval()
		accuracy = (model(test_x)[:,-max_delay:,:].argmax(-1).T == test_y)
		accuracy = accuracy.cpu().numpy().astype(float)

	for i, delay in enumerate(delays):
		p = accuracy[delay-max_delay,:].mean()
		std = np.sqrt(p * (1-p) / N) # Standard error of the mean
		accuracies_mean[i] = p*100
		accuracies_std[i] = std*100
		# print(f"{delay}: {accuracy.mean()}")
	return delays, accuracies_mean, accuracies_std

def plot_accuracies(directory, model_list, exclude_list):
	accuracies = np.empty((len(model_list), MAX_DELAY))
	for i, model_dir in enumerate(model_list):
		if model_dir in exclude_list:
			continue
		try:
			model = retrieve_model(directory + model_dir).to(DEVICE)
		except:
			continue

		delays, accuracies_mean, accuracies_std = evaluate_delays(model, test_x, test_y, max_delay)
		# accuracies_mean /= count_trainable_parameters(model)

		n = get_n_neurons(model)
		d = get_n_dendrites(model)
		if d == None:
			d = 1
			ls = ":"
		else:
			ls = "-"

		name_str = names_dict[type(model).__name__] + f"_{n}_{d}"
		line, = plt.plot(
			RELATIVE_DELAY,
			accuracies_mean,
			# alpha=1 - 1/(2*d),
			# c='C1',
			ls = ls,
			label=name_str
			)
		# plt.fill_between(relative_delay, accuracies_mean + accuracies_std, accuracies_mean - accuracies_std,
		# 		   label=name_str, facecolor=line.get_color(), ls='--',
		# 		   alpha=.2)

	return accuracies



if __name__ == '__main__':
	from config.ray.pbt import CONFIG
	torch.manual_seed(22)

	CONFIG['type'] = 'gauss'
	CONFIG['id'] = 'debug'
	CONFIG['energy'] = 1.
	CONFIG['sigma'] = 1.
	CONFIG['gamma'] = 1.
	CONFIG['learning_rate'] = 1e-5
	CONFIG['max_epochs'] = 1
	CONFIG['steps_per_epoch'] = 1
	CONFIG['batch_size'] = 1




#%%
	with plt.style.context('dark'):
		max_delay = MAX_DELAY
		n_samples = N
		baseline_list = [
			"37767",
			# "41343",
			# "42914",
			]
		model_list = [
			"43842",
			"43693",
			"43207",
			"43230",
			"43545",
		]
		exclude_list = [
			"42391",
		]

		model_directory = '/Users/cankayser/Downloads/results/'
		baseline_directory = '/Users/cankayser/Downloads/results_baseline/'
		# model_list = list(listdir(model_directory))


		names_dict = {
			"LIF_SNN": "LIF",
			"NMDA_AMPA_LIF_SNN": "NMDA",
			"NMDA_AMPA_GABA_LIF_SNN": "NMDA",
			"NMDA_READOUT_SNN": "NMDA Readout",
		}

		test_loader = build_test_loader(N, delay_steps=max_delay)

		test_x, test_y = next(iter(test_loader))
		test_x = test_x.to(DEVICE)
		test_y = test_y.to(DEVICE)
		T_max = test_x.shape[1]
		w, h = plt.rcParams['figure.figsize']
		fig, ax = plt.subplots(figsize=(w,h))

		delays = np.arange(0,max_delay)
		RELATIVE_DELAY = ((delays / T_max)) #* 100

		plot_accuracies(baseline_directory, baseline_list, exclude_list)
		accuracies = plot_accuracies(model_directory, model_list, exclude_list)
		# mean = accuracies.mean(axis=0)
		# min = accuracies.min(axis=0)
		# max = accuracies.max(axis=0)
		# std = accuracies.std(axis=0)
		# plt.plot(relative_delay, mean+(2*std), c='C1')
		# plt.plot(relative_delay, max, c='C1')
		# plt.plot(relative_delay, mean, c='C1', label='NMDA mean')
		# plt.plot(relative_delay, min, c='C1')
		# plt.plot(relative_delay, mean-(2*std), c='C1')

		plt.plot(RELATIVE_DELAY, RELATIVE_DELAY*0 + 5, ls=':', c='grey', label='Guessing Accuracy')
		plt.xlabel(r'Delay Period / Sequence Length [1]')
		plt.ylabel(r'Accuracy [$\%$]')
		plt.ylim(35,65)
		# plt.yscale('log')
		plt.legend()
		plt.savefig('/Users/cankayser/Downloads/evaluate_delays')
		plt.show()



























		# %%
