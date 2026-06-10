#%%
import torch
import torch.nn as nn

from utils.settings import DEVICE
from utils.surrogate import Surrogate
from neurons import NeuronLayer, LinearReadoutLayer

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


	def forward(self, x, collect_states=False):
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
		return self.forward(x, collect_states=True)




class DendriticSNN(SNN):
	def __init__(self, config):
		super().__init__()

		self.layer_list = [
			NeuronLayer(config['n_inputs'], config['n_dendrites'], config['n_hidden'], config),
			NeuronLayer(config['n_hidden'], config['n_dendrites'], config['n_hidden'], config),
			LinearReadoutLayer(config['n_hidden'], config['n_outputs'], config),
		]

		self.build()


# %%


ARCHITECTURES = {
	'DendriticSNN': DendriticSNN,
}

if __name__ == "__main__":
	from config.ray.pbt import CONFIG
	from datasets.shd_prediction import build_loader
	from utils.diagnosis import count_trainable_parameters
	import matplotlib.pyplot as plt

	CONFIG['seed'] = 1234
	CONFIG['type'] = 'gauss'
	CONFIG['id'] = 'debug'
	CONFIG['energy'] = 1.
	CONFIG['sigma'] = 1.
	CONFIG['gamma'] = 1.
	CONFIG['dt'] = 1e-5
	CONFIG['learning_rate'] = 1e-3
	CONFIG['max_epochs'] = 1
	CONFIG['steps_per_epoch'] = 1
	CONFIG['batch_size'] = 3
	CONFIG['learnable'] = 'none'

	CONFIG['surrogate_spike'] = Surrogate()
	loader = build_loader(1)
	x, y = next(iter(loader))
	x = x.to(DEVICE)
	model = DendriticSNN(CONFIG).to(DEVICE)
	p = count_trainable_parameters(model)
	print(p)
	model.test(x*9999)
	# plt.imshow(
	# 	model(
	# 		x*99999999
	# 	)[0].detach().cpu()
	# )
# %%
