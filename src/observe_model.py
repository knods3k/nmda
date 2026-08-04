#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from retrieve_model import retrieve_model
from ray_training import build_loader, build_test_loader, build_model
from utils.settings import DEVICE
from utils.nmda_init import initialise_nmda_weights
from config.ray.pbt import CONFIG
# %%
N = 2264
N = N // 100

dir = '/Users/cankayser/Downloads/ray_results/50292'




CONFIG['seed'] = 123
CONFIG['type'] = 'gauss'
CONFIG['id'] = 'debug'
CONFIG['energy'] = 1.
CONFIG['sigma'] = 1.
CONFIG['gamma'] = 1.
CONFIG['dt'] = 1e-2
CONFIG['learning_rate'] = 1e-2
CONFIG['max_epochs'] = 1
CONFIG['steps_per_epoch'] = 1
CONFIG['batch_size'] = 3
CONFIG['learnable'] = 'none'
CONFIG['noise_variance'] = 0
CONFIG['architecture'] = 'DendriticSNN_Affine'

CONFIG['n_hidden'] = 64
CONFIG['n_dendrites'] = 8
CONFIG['dendritic_scaling'] = 50

torch.manual_seed(CONFIG['seed'])
max_delay = 199
test_loader = build_loader(N, delay_steps=max_delay)

# model = retrieve_model(dir).to(DEVICE)
model = build_model(CONFIG).to(DEVICE)
initialise_nmda_weights(model)



test_x, test_y = next(iter(test_loader))
test_x = test_x.to(DEVICE)
test_y = test_y.to(DEVICE)


states = model.test(test_x)

#%%
for i, layer in enumerate(model.layers):
	for state_name, state in states[i].items():
		if i == len(model.layers) - 1:
			state = torch.softmax(state, dim=-1)
		plt.imshow(state[0].detach().cpu().T, interpolation='None', cmap='berlin', vmin=-1, vmax=1)
		# plt.plot(state[0].detach().cpu().T)
		plt.show()
	# if i == 0:
		# break





# %%
