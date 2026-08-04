#%%
from ray_training import build_loader, \
	build_test_loader, build_surrogate, build_model, build_optimizer, build_evaluation_function,\
	train
from utils.settings import DEVICE
from utils.nmda_init import initialise_nmda_weights
from neurons import NeuronModel
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	from config.ray.pbt import CONFIG


	CONFIG['seed'] = 4
	CONFIG['type'] = 'gauss'
	CONFIG['id'] = 'debug'
	CONFIG['energy'] = 1.
	CONFIG['sigma'] = 1.
	CONFIG['gamma'] = 1.
	CONFIG['learning_rate'] = 1e-5
	CONFIG['max_epochs'] = 1
	CONFIG['steps_per_epoch'] = 1
	CONFIG['batch_size'] = 1
	CONFIG['learnable'] = 'all'

	CONFIG['dt'] = 1e-1
	CONFIG['relative_concentration'] = 1
	CONFIG['n_hidden'] = 4

	# with torch.autograd.detect_anomaly():
	# 	train(CONFIG)

	torch.manual_seed(CONFIG['seed'])
	config = CONFIG
	loader = build_loader(config['batch_size'])
	test_loader = build_test_loader(2264)

	surrogate = build_surrogate(config)
	model = build_model(config, surrogate)
	initialise_nmda_weights(model, loader)
	optimizer = build_optimizer(config, model=model)
	evaluate = build_evaluation_function(config)

	from utils.diagnosis import count_trainable_parameters
	count_trainable_parameters(model)
	x,y = next(iter(loader))
	x = x.to(DEVICE)
	model.to(DEVICE)
	model(x)


def observe_states(model, x, layer_idx, state_str):

	with torch.no_grad():
		if isinstance(model.layers[layer_idx], NeuronModel):
			states = model.test(x)
			T = x.shape[1]
			N = states[0][layer_idx][state_str].shape[1]
			s = np.empty((T,N))
			for i, state in enumerate(states):
				s[i] = state[layer_idx][state_str].cpu().numpy()

			return s
		else:
			return None

states = []
names = []
n_layers = len(model.layers)
for i in range(n_layers):
	state = observe_states(model, x, i, 'u')
	if state is not None:
		states.append(state.copy())
		names.append(type(model.layers[i]).__name__)


n_states = len(states)
plt.subplots(n_states, 1, figsize=(8,1.5*n_states), sharex=True)
for i, state in enumerate(states):
	plt.subplot(n_states, 1, i+1)
	plt.plot(state)
	plt.title(names[i])
	plt.ylabel('Potential [V]')
plt.xlabel('Time [s]')
plt.show()
# %%

# %%


# %%
