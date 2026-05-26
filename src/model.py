#%%
import torch
import torch.nn as nn

from utils.settings import DEVICE
from utils.surrogate import Surrogate
from neurons import LIF, LI, NMDA,  NMDA_AMPA, NMDA_AMPA_ablation, NeuronModel, DendriteModel, \
	NonNegativeLinear, COMPARTMENTS, NMDA_AMPA_GABA

N_HIDDEN = 256

class DegenerateBaseline(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.degenerate = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

	def forward(self, input_data):
		return 1.0*input_data*self.degenerate


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
			if isinstance(layer, NeuronModel):
				self.neuron_models.append(layer)

	def forward(self, x):
		B, T, D = x.shape

		for neuron in self.neuron_models:
			neuron.reset(B)
			neuron.prepare()

		out = []

		for t in range(T):
			x_t = x[:, t, :]

			for layer in self.layers:
				x_t = layer(x_t)

			out.append(x_t)

		return torch.stack(out, dim=1)

	def test(self, x):
		B, T, D = x.shape

		for neuron in self.neuron_models:
			neuron.reset(B)

		out = []
		states = []

		for t in range(T):
			x_t = x[:, t, :]

			layer_state_list = []
			layer_names = []
			for layer in self.layers:
				x_t = layer(x_t)
				layer_name = type(layer).__name__
				layer_names.append(layer_name)
				if isinstance(layer, NeuronModel):
					layer_state_list.append({k: v.clone() for k, v in layer.state.items()})
				else:
					layer_state_list.append(None)

			out.append(x_t)
			states.append(layer_state_list)

		return states




class NMDA_AMPA_LIF_SNN(SNN):
	def __init__(self,config):
		super().__init__()


		self.layer_list = [
			NMDA_AMPA(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NMDA_AMPA(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			nn.Linear(
				config['n_hidden'],
				config['n_outputs']
				),
			LI(
				config['n_outputs'],
				config),
		]

		self.build()

class NMDA_AMPA_LIF_SNN_fix_omega(NMDA_AMPA_LIF_SNN):
	def __init__(self, config):
		super().__init__(config)
		for layer in self.layer_list:
			if isinstance(layer, DendriteModel):
				layer.w.requires_grad = False


class NMDA_AMPA_GABA_LIF_SNN(SNN):
	def __init__(self,config):
		super().__init__()


		self.layer_list = [
			NMDA_AMPA_GABA(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NMDA_AMPA_GABA(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			nn.Linear(
				config['n_hidden'],
				config['n_outputs']
				),
			LI(
				config['n_outputs'],
				config),
		]

		self.build()

class NMDA_AMPA_GABA_LIF_SNN_readout_ablation(NMDA_AMPA_GABA_LIF_SNN):
	def __init__(self, config):
		super().__init__(config)
		for layer in self.layer_list[:-2]:
			for p in layer.parameters():
				p.requires_grad = False


class NMDA_AMPA_GABA_LIF_SNN_linear_ablation(NMDA_AMPA_GABA_LIF_SNN):
	@staticmethod
	def h(u, gam0=.5, gam1=8.):
		return 1.

	def __init__(self, config):
		super().__init__(config)
		for layer in self.layer_list:
			if isinstance(layer,NMDA):
				layer.h = self.h


class NMDA_READOUT_SNN(SNN):
	def __init__(self,config):
		super().__init__()


		self.layer_list = [
			NMDA_AMPA(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NMDA_AMPA(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NMDA_AMPA(
				config['n_hidden'],
				config['n_outputs']*config['n_dendrites'],
				config),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
		]

		self.build()


class NMDA_ablation(SNN):
	def __init__(self,config):
		super().__init__()


		self.layer_list = [
			NonNegativeLinear(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites']
				),
			NMDA_AMPA_ablation(
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NonNegativeLinear(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites']
				),
			NMDA_AMPA_ablation(
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NonNegativeLinear(
				config['n_hidden'],
				config['n_outputs']
				),
			LI(
				config['n_outputs'],
				config),
		]

		self.build()


class NMDA_LIF_SNN(SNN):
	def __init__(self, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.layer_list = [
			NonNegativeLinear(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites']
				),
			NMDA(
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NonNegativeLinear(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites']
				),
			NMDA(
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NonNegativeLinear(
				config['n_hidden'],
				config['n_outputs']
				),
			LI(
				config['n_outputs'],
				config),
		]

		self.build()

class LI_LIF_SNN(SNN):
	def __init__(self, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.layer_list = [
			NonNegativeLinear(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites']
				),
			LI(
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NonNegativeLinear(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites']
				),
			LI(
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			NonNegativeLinear(
				config['n_hidden'],
				config['n_outputs']
				),
			LI(
				config['n_outputs'],
				config),
		]

		self.build()

class LIF_SNN(SNN):
	def __init__(self, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.layer_list = [
			nn.Linear(config['n_inputs'], config['n_hidden']*config['n_dendrites']),
			LIF(config['n_hidden']*config['n_dendrites'], config),
			nn.Linear(config['n_hidden']*config['n_dendrites'], config['n_hidden']),
			LIF(config['n_hidden'], config),
			nn.Linear(config['n_hidden'], config['n_outputs']),
			LI(config['n_outputs'], config),
		]

		self.build()



class NMDA_AMPA_COMP_LIF_SNN(SNN):
	def __init__(self,config):
		super().__init__()


		self.layer_list = [
			COMPARTMENTS(
				config['n_inputs'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			COMPARTMENTS(
				config['n_hidden'],
				config['n_hidden']*config['n_dendrites'],
				config
				),
			DendriteModel(
				config['n_dendrites'],
				config,
				),
			LIF(
				config['n_hidden'],
				config
				),
			nn.Linear(
				config['n_hidden'],
				config['n_outputs']
				),
			LI(
				config['n_outputs'],
				config),
		]

		self.build()


# %%


ARCHITECTURES = {
	'NMDA_LIF_SNN': NMDA_LIF_SNN,
	'NMDA_AMPA_LIF_SNN': NMDA_AMPA_LIF_SNN,
	'NMDA_AMPA_LIF_SNN_fix_omega': NMDA_AMPA_LIF_SNN_fix_omega,
	'NMDA_AMPA_GABA_LIF_SNN': NMDA_AMPA_GABA_LIF_SNN,
	'NMDA_AMPA_GABA_LIF_SNN_readout_ablation': NMDA_AMPA_GABA_LIF_SNN_readout_ablation,
	'NMDA_AMPA_GABA_LIF_SNN_linear_ablation': NMDA_AMPA_GABA_LIF_SNN_linear_ablation,
	'NMDA_AMPA_COMP_LIF_SNN': NMDA_AMPA_COMP_LIF_SNN,
	'NMDA_ablation': NMDA_ablation,
	'LI_LIF_SNN': LI_LIF_SNN,
	'LIF_SNN': LIF_SNN,
	'NMDA_READOUT_SNN': NMDA_READOUT_SNN,
}

if __name__ == "__main__":
	from config.ray.pbt import CONFIG
	from datasets.shd_prediction import build_loader
	from utils.diagnosis import count_trainable_parameters

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
	model = NMDA_AMPA_LIF_SNN(CONFIG).to(DEVICE)
	p = count_trainable_parameters(model)

	model = NMDA_AMPA_GABA_LIF_SNN_readout_ablation(CONFIG).to(DEVICE)
	q = count_trainable_parameters(model)
	print(p-q)
	model(x)
# %%
