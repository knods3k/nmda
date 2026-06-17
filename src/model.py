#%%
import torch
import torch.nn as nn

from utils.settings import DEVICE
from utils.surrogate import Surrogate
from neurons import DendriteLayer, SomaLayer, LinearReadoutLayer, CoupledDendriticLayer, AdditiveLayer, MultiplicativeLayer, AffineLayer

N_HIDDEN = 256


class SNN(nn.Module):
	def __init__(self,
			  layer_list=(),
			  *args,
			  **kwargs):
		super().__init__(*args, **kwargs)

		self.layer_list = layer_list
		self.neuron_models = []
		self.build()

	def build(self):
		self.layers = nn.ModuleList()
		for layer in self.layer_list:
			self.layers.append(layer)


	def forward(self, x):
		B, T, D = x.shape

		for layer in self.layers:
			layer.reset(B)

		out = []

		for t in range(T):
			x_t = x[:, t, :]

			for layer in self.layers:
				x_t = layer(x_t)

			out.append(x_t)

		return torch.stack(out, dim=1)

	def test(self, x):
		B, T, D = x.shape

		for layer in self.layers:
			layer.reset(B)

		states = [{} for _ in self.layers]
		for i, layer in enumerate(self.layers):
			for key, value in layer.state.items():
				states[i][key] = []

		for t in range(T):
			x_t = x[:, t, :]

			for i, layer in enumerate(self.layers):
				x_t = layer(x_t)
				for key, value in layer.state.items():
					states[i][key].append(value)


		for i, layer in enumerate(self.layers):
			for key, value in layer.state.items():
				states[i][key] = torch.stack(states[i][key], dim=1).clone().detach()

		return states



class DendriticSNN(SNN):
	def __init__(self, config):
		super().__init__()

		n_in = config['n_inputs']
		n_dendrites = config['n_dendrites']
		n_hidden = config['n_hidden']
		n_out = config['n_outputs']

		self.layer_list = [
			DendriteLayer(n_in, n_dendrites, n_hidden, config),
			SomaLayer(n_hidden, config),
			DendriteLayer(n_hidden, n_dendrites, n_hidden, config),
			SomaLayer(n_hidden, config),
			LinearReadoutLayer(n_hidden, n_out, config),
		]

		self.build()


class DendriticSNN_coupled(SNN):
	def __init__(self, config, coupling_class):
		super().__init__()

		n_in = config['n_inputs']
		n_dendrites = config['n_dendrites']
		n_hidden = config['n_hidden']
		n_out = config['n_outputs']

		self.layer_list = [
			CoupledDendriticLayer(n_in, n_dendrites, n_hidden, coupling_class,  config),
			SomaLayer(n_hidden, config),
			CoupledDendriticLayer(n_hidden, n_dendrites, n_hidden, coupling_class, config),
			SomaLayer(n_hidden, config),
			LinearReadoutLayer(n_hidden, n_out, config),
		]

		self.build()

class DendriticSNN_Additive(DendriticSNN_coupled):
	def __init__(self, config):
		super().__init__(config, AdditiveLayer)

class DendriticSNN_Multiplicative(DendriticSNN_coupled):
	def __init__(self, config):
		super().__init__(config, MultiplicativeLayer)

class DendriticSNN_Affine(DendriticSNN_coupled):
	def __init__(self, config):
		super().__init__(config, AffineLayer)


# %%


ARCHITECTURES = {
	'DendriticSNN': DendriticSNN,
	'DendriticSNN_Additive': DendriticSNN_Additive,
	'DendriticSNN_Multiplicative': DendriticSNN_Multiplicative,
	'DendriticSNN_Affine': DendriticSNN_Affine,
}

if __name__ == "__main__":
	from config.ray.pbt import CONFIG
	from datasets.shd_prediction import build_loader
	from utils.diagnosis import count_trainable_parameters
	import matplotlib.pyplot as plt
	from utils.nmda_init import initialise_nmda_weights

	CONFIG['seed'] = 1234
	CONFIG['type'] = 'gauss'
	CONFIG['id'] = 'debug'
	CONFIG['energy'] = 1.
	CONFIG['sigma'] = 1.
	CONFIG['gamma'] = 1.
	CONFIG['learning_rate'] = 1e-3
	CONFIG['max_epochs'] = 1
	CONFIG['steps_per_epoch'] = 1
	CONFIG['batch_size'] = 3
	CONFIG['learnable'] = 'none'

	CONFIG['surrogate_spike'] = Surrogate()
	loader = build_loader(1)
	x, y = next(iter(loader))
	x = x.to(DEVICE)
	model = DendriticSNN_Affine(CONFIG).to(DEVICE)
	initialise_nmda_weights(model)
	p = count_trainable_parameters(model)
	print(p)
	states = model.test(x*9)
	u = states[0]['u'].cpu()
	plt.imshow(u.detach().cpu()[0].T, vmin=-1, vmax=1, cmap='berlin')
	plt.colorbar()
	# plt.imshow(
	# 	model(
	# 		x*99999999
	# 	)[0].detach().cpu()
	# )



# %%
