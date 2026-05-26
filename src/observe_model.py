#%%
from ray_training import build_test_loader, build_surrogate, build_optimizer, build_evaluation_function, build_model, build_loader
from utils.settings import DEVICE
from neurons import NeuronModel, LIF, LI, DendriteModel
import torch
import numpy as np

import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import gaussian_kde
from utils.nmda_init import initialise_nmda_weights, initialise_nmda_weights_lognormal
from retrieve_model import retrieve_model

#%%
dir = "/Users/cankayser/Downloads/results/43230"
# dir = ''

DELAY = 200
B = 1


def observe_states(model, x, layer_idx, state_str):

	with torch.no_grad():
		layer = model.layers[layer_idx]
		if isinstance(layer, NeuronModel):
			if isinstance(layer, LIF):
				state_str = 's'

			states = model.test(x)
			T = x.shape[1]
			N = states[0][layer_idx][state_str].shape[1]
			s = np.empty((B,T,N))
			for i, state in enumerate(states):
				s_hat = state[layer_idx][state_str]
				s[:,i,:] = s_hat.cpu().numpy()

			return s
		else:
			return None

def list_states(model):
	states = []
	names = []
	n_layers = len(model.layers)
	for i in range(n_layers):
		state = observe_states(model, test_x, i, 'u')
		if state is not None:
			states.append(state.copy())
			names.append(type(model.layers[i]).__name__)

	return names, states


if __name__ == '__main__':
	from config.ray.pbt import CONFIG

	# CONFIG['seed'] = 1234 # failure
	CONFIG['seed'] = 1234
	CONFIG['type'] = 'gauss'
	CONFIG['id'] = 'debug'
	CONFIG['energy'] = 1.
	CONFIG['sigma'] = 1.
	CONFIG['gamma'] = 1.
	CONFIG['learning_rate'] = 1e-5
	CONFIG['max_epochs'] = 1
	CONFIG['steps_per_epoch'] = 1
	CONFIG['batch_size'] = 1
	CONFIG['architecture'] = 'NMDA_AMPA_GABA_LIF_SNN'
	# CONFIG['learnable'] = 'none'

	names_dict = {
		"NMDA_AMPA_GABA": "Potential $v$",
		"NMDA_AMPA": "Potential $v$",
		"LIF": "Spikes $z$",
		"LI": "Readout $u$",
	}
	torch.manual_seed(CONFIG['seed'])
	config = CONFIG

	surrogate = build_surrogate(config)
	if dir == '':
		model = build_model(config, surrogate)
		initialise_nmda_weights(model)
	else:
		model = retrieve_model(dir)

	optimizer = build_optimizer(config, model=model)
	evaluate = build_evaluation_function(config)


# %%

	#%%
	with plt.style.context('dark'):
		test_loader = build_test_loader(
			B,
			delay_steps=DELAY,
			)
		test_x, test_y = next(iter(test_loader))
		test_x = test_x.to(DEVICE)

		model.to(DEVICE)
		model(test_x)

		names, states = list_states(model)
		n_states = len(states)
		w, h = plt.rcParams['figure.figsize']
		fig, ax = plt.subplots(n_states, 1, figsize=(w,h*1.5), sharex=True, layout='compressed')
		for i, state in enumerate(states):
			plt.subplot(n_states, 1, i+1)

			if i+1 == n_states:
				state = softmax(state, axis=-1)
				state.T[int(test_y),::19] = np.nan

				state[:,-DELAY,::2] = np.nan

			# if np.any(state < 0):
			# 	# cmap = plt.cm.managua_r
			# 	cmap = plt.cm.grey

			# else:
			# 	cmap = plt.cm.grey

			if 0 <= i < 1:
				plt.title('Input Layer')
			if 2 <= i < 3:
				plt.title('Hidden Layer')
			if 4 <= i < 5:
				plt.title('Readout Layer')

			cmap = plt.cm.managua_r

			cmap.set_bad('red')
			im = plt.imshow(state.T, origin=r'lower', cmap=cmap, vmin=-1, vmax=1)

			# plt.imshow(state.T, origin=r'lower')
			# plt.ylabel(rf"{names_dict[names[i]]}" + rf" (Layer {(i//2)+1})")
			plt.ylabel(rf"{names_dict[names[i]]}")
			plt.yticks([])
			# plt.ylabel(r'Subunit')


		plt.colorbar(label=r'Potential [V]', ax=ax, location='top', fraction=.05)
		plt.xlabel(r'Time [ms]')
			# plt.vlines(state[0]-DELAY, c='r', ls=':')
		# plt.savefig(f'/Users/cankayser/Downloads/neural_dynamics')
		plt.show()



#%%
if __name__ == '__main__':
	DELAY = 0
	B = 19
	test_loader = build_test_loader(
		B,
        delay_steps=DELAY,
		)
	test_x, test_y = next(iter(test_loader))
	test_x = test_x.to(DEVICE)

	model = retrieve_model(dir)
	model.to(DEVICE)
	model(test_x)
	names, states_post = list_states(model)


	initialise_nmda_weights(model)
	model.to(DEVICE)
	model(test_x)
	names, states_pre = list_states(model)

#%%
	def plot_histogram(data, bins='sqrt', **kw):
		counts, edges = np.histogram(data, bins, density=True)
		centers = .5 * (edges[:-1] + edges[1:])
		plt.plot(centers, counts, linewidth=1., **kw)
		# plt.hist(data, histtype='step', align='mid', density=True, **kw)

	with plt.style.context('save'):
		plt.subplots(2,1, sharex=True, layout='compressed')
		x = np.linspace(-1,1,999)

		plt.subplot(211)
		plt.title(r'Input Layer')
		plot_histogram(states_pre[0].flatten(), label=r'Before Training')
		plot_histogram(states_post[0].flatten(), label=r'After Training')
		# kde = gaussian_kde(states_pre[0].flatten())
		# plt.plot(x, kde.pdf(x), label='Input Layer')
		# kde = gaussian_kde(states_pre[2].flatten())
		# plt.plot(x, kde.pdf(x), label='Hidden Layer')
		plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
		plt.ylabel(r"Count [$1$]")
		plt.ylim(0,9)
		plt.yticks([])

		plt.subplot(212)
		plt.title(r'Hidden Layer')
		plot_histogram(states_pre[2].flatten(), label=r'Before Training')
		plot_histogram(states_post[2].flatten(), label=r'After Training')
		# kde = gaussian_kde(states_post[0].flatten())
		# plt.plot(x, kde.pdf(x), label='Input Layer')
		# kde = gaussian_kde(states_post[2].flatten())
		# plt.plot(x, kde.pdf(x), label='Hidden Layer')
		plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
		plt.ylabel(r"Count [$1$]")
		plt.legend(loc='upper left')

		plt.xlabel(r"Potential $u$")
		plt.xlim((-1.1,1.1))
		plt.ylim(0,9)
		plt.yticks([])

		plt.savefig('/Users/cankayser/Downloads/state_histogram')
		plt.show()





# %%
