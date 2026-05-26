#%%
from os import listdir

import torch
import numpy as np
import matplotlib.pyplot as plt

from ray_training import build_test_loader
from retrieve_model import retrieve_model
from utils.settings import DEVICE
from utils.diagnosis import get_n_neurons, get_n_dendrites, count_trainable_parameters

N = 2264 // 10
MAX_NOISE = .1
N_STEPS = 9
noise_rates = np.linspace(0,MAX_NOISE,N_STEPS)
SNR = noise_rates / MAX_NOISE

#%%
test_loader = build_test_loader(N)

test_x, test_y = next(iter(test_loader))
test_x = test_x.to(DEVICE)
test_y = test_y.to(DEVICE)


B, T, D = test_x.shape
noise_npy = np.ones((1,T,D)) * noise_rates[:,None,None,None]
noise = torch.tensor(
		np.random.poisson(noise_npy
	),
	device=DEVICE,
)

TEST_X_NOISE = (test_x.unsqueeze(0) + noise).view(N_STEPS*B, T, D)
TEST_Y_NOISE = test_y.unsqueeze(0)
#%%


def evaluate_noise(model, test_x_noise, test_y_noise):
	accuracies = np.zeros(noise.shape[0], dtype=float)
	with torch.no_grad():
		model.eval()
		logits = model(test_x_noise)[:,-1,:]
		predictions = logits.argmax(-1).view(N_STEPS, B)
		accuracies = (predictions == test_y_noise).float().mean(dim=1) * 100
		accuracies = accuracies.cpu().numpy().astype(float)
	return accuracies


def plot_noise(directory, model_list, exclude_list):
	accuracies = np.empty((len(model_list), N_STEPS))
	for i, model_dir in enumerate(model_list):
		if model_dir in exclude_list:
			continue
		try:
			model = retrieve_model(directory + model_dir).to(DEVICE)
		except:
			continue

		accuracies_mean = evaluate_noise(model, TEST_X_NOISE, TEST_Y_NOISE)
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
			SNR,
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




	with plt.style.context('save'):
		max_noise = MAX_NOISE
		n_samples = N
		baseline_list = [
			"37767",
			"41343",
			"42914",
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


		names_dict = {
			"LIF_SNN": "LIF",
			"NMDA_AMPA_LIF_SNN": "NMDA",
			"NMDA_AMPA_GABA_LIF_SNN": "NMDA",
			"NMDA_READOUT_SNN": "NMDA Readout",
		}

		test_loader = build_test_loader(N)

		test_x, test_y = next(iter(test_loader))
		test_x = test_x.to(DEVICE)
		test_y = test_y.to(DEVICE)
		T_max = test_x.shape[1]
		w, h = plt.rcParams['figure.figsize']
		plt.figure(figsize=(w,h))

		accuracies = plot_noise(baseline_directory, baseline_list, exclude_list)
		accuracies = plot_noise(model_directory, model_list, exclude_list)
		mean = accuracies.mean(axis=0)
		min = accuracies.min(axis=0)
		max = accuracies.max(axis=0)
		std = accuracies.std(axis=0)
		# plt.plot(relative_delay, mean+(2*std), c='C1')
		# plt.plot(relative_delay, max, c='C1')
		# plt.plot(relative_delay, mean, c='C1', label='NMDA mean')
		# plt.plot(relative_delay, min, c='C1')
		# plt.plot(relative_delay, mean-(2*std), c='C1')
		plt.plot(SNR, SNR*0 + 5, ls=':', c='grey', label='Guessing Accuracy')
		# plt.xticks(noise_rates[::10], noise_rates[::10]/.2)
		plt.xlabel(r'Signal-to-Noise Ratio [1]')
		plt.ylabel(r'Accuracy [$\%$]')
		plt.ylim(0,100)
		# plt.yscale('log')
		plt.legend()
		plt.savefig('/Users/cankayser/Downloads/evaluate_noise')
		# plt.show()




			# %%
