#%%
import torch
import torch.nn as nn

from utils.settings import DEVICE, MACHINE_EPSILON, DTYPE


class BiologicalModel(nn.Module):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.n_neurons = n_neurons
		self.state = {}

	def register_logarithmic_parameters(
			self,
			parameter_dictionary: dict,
			learnable: str ='none',
			initial_parameter_spread: float = 0.01,
			device=DEVICE,
			dtype=DTYPE):

		for name, value in parameter_dictionary.items():
			value = torch.log(torch.tensor(value, device=device, dtype=dtype))
			if learnable == 'all':
				param = nn.Parameter(value + initial_parameter_spread*torch.randn((self.n_neurons), device=device, dtype=dtype))
			elif learnable == 'one':
				param = nn.Parameter(value*torch.ones(1, device=device, dtype=dtype))
			elif learnable == 'none':
				param = nn.Parameter(value*torch.ones(1, device=device, dtype=dtype), requires_grad=False)
			else:
				raise ValueError(f'Unkown learnable mode {learnable}. Set to one of [\'all\', \'one\', \'none\']')

			setattr(self, name, param)


	def reset(self, batch_size):
		for key in self.state.keys():
			self.state[key] = torch.zeros((batch_size, self.n_neurons), device=DEVICE, dtype=DTYPE)
		return self.state


class DendriticSummation(nn.Module):
	def __init__(self, n_dendrites, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.n_dendrites = n_dendrites

		if 'dendritic_scaling' in config:
			winit = config['dendritic_scaling']
			self.w = nn.Parameter(torch.tensor(winit), requires_grad=False)
		else:
			winit = 1.
			self.w = nn.Parameter(torch.tensor(winit), requires_grad=True)

	def forward(self, x):
		B, D = x.shape
		return (x.view((B, -1, self.n_dendrites)) * self.w ).mean(-1)



class NonNegativeLinear(nn.Module):
	def __init__(self, in_features, out_features, config):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=DTYPE, device=DEVICE))
		if config['activate_bias']:
			self.bias = nn.Parameter(torch.randn(out_features, dtype=DTYPE, device=DEVICE))
		else:
			self.bias = torch.zeros(1, dtype=DTYPE, device=DEVICE)

	def forward(self, input):
		return nn.functional.linear(input, self.weight.abs(), self.bias.abs())


class ExponentialDecayFilter(nn.Module):
	def __init__(self, du_log, dt_log, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.dt = torch.exp(dt_log)
		self.rate = torch.exp(du_log)
		self.inv = (1 / self.rate)
		self.decay = torch.exp(-self.rate * self.dt)
		self.drive = -torch.expm1(-self.rate * self.dt)

	def __call__(self, u, i):
		return (self.decay * u) + (self.drive * self.inv * i)


class LI(BiologicalModel):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(n_neurons, config, *args, **kwargs)

		self.state = {'i': None, 'u': None}

		self.register_logarithmic_parameters(
			{
				'dt_log': config['dt'],
				'du_soma_log': config['du_soma'],
			},
			learnable=config['learnable']
		)

		self.filter = ExponentialDecayFilter(self.du_soma_log, self.dt_log)


	def forward(self, x):

		self.state['i'] = x

		u = self.state['u']
		i = self.state['i']

		u_new = self.filter(u, i)

		self.state['u'] = u_new
		return self.state['u']


class LIF(LI):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(n_neurons, config, *args, **kwargs)

		self.state = {'i': None, 'u': None, 's': None}

		self.threshold = config['threshold']
		self.surrogate_spike = config['surrogate_spike']


	def forward(self, x):

		self.state['i'] = x

		u = self.state['u']
		i = self.state['i']
		s = self.surrogate_spike((u - self.threshold))

		u_hat = self.filter(u,i)
		u_new = u_hat * (1 - s)

		self.state['u'] = u_new
		self.state['s'] = s
		return s


class NMDA_Receptor(LI):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(n_neurons, config, *args, **kwargs)

		self.state = {'i': None, 'w': None,  'v': None}

		self.register_logarithmic_parameters(
			{
				'du_dend_log': config['du_dend'],
				'dv_log': config['dv'],
				'dw_log': config['dw'],
			},
			learnable=config['learnable']
		)

		self.gam0 = config['gam0']
		self.gam1 = config['gam1']

		self.filter_v = ExponentialDecayFilter(self.dv_log, self.dt_log)
		self.filter_w = ExponentialDecayFilter(self.dw_log, self.dt_log)

		self.dw = self.filter_w.rate
		self.dv = self.filter_v.rate

		self.norm = -torch.exp(-self.dw*torch.log(self.dv/self.dw)/(self.dv - self.dw)) + torch.exp(-self.dv*torch.log(self.dv/self.dw)/(self.dv - self.dw))


	@staticmethod
	def h(u, gam0=.5, gam1=8.):
		return torch.sigmoid(gam1 * (u - gam0))


	def conductance(self, i, u):
		w = self.state['w']
		v = self.state['v']

		v_new = self.filter_v(v,i)
		w_new = self.filter_w(w,i)

		g_nmda = self.norm * (v_new - w_new) * self.h(u, gam0=self.gam0, gam1=self.gam1)

		self.state['v'] = v_new
		self.state['w'] = w_new

		return g_nmda


class AMPA_Receptor(LI):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(n_neurons, config, *args, **kwargs)

		self.state = {'i': None, 'g': None}

		self.register_logarithmic_parameters(
			{
				'du_dend_log': config['du_dend'],
				'dg_log': config['dg'],
			},
			learnable=config['learnable']
		)

		self.gam0 = config['gam0']
		self.gam1 = config['gam1']

		self.filter = ExponentialDecayFilter(self.dg_log, self.dt_log)

	def conductance(self, i):
		g = self.state['g']
		g_new = self.filter(g,i)

		return g_new

class GABA_Receptor(AMPA_Receptor):
	pass


class MembraneIntegrator(nn.Module):
	def __init__(self, du, dt, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.du = du
		self.dt = dt

	def integrate(self, u, excitation, inhibition):
		return (u + self.dt * (excitation - inhibition)) / (1 + self.dt * (self.du + excitation + inhibition))



class DendriteLayer(BiologicalModel):
	def __init__(self, n_inputs, n_dendrites, n_outputs, config, *args, **kwargs):
		super().__init__(n_dendrites * n_outputs, config, *args, **kwargs)

		self.nmda = NMDA_Receptor(n_dendrites * n_outputs, config)
		self.ampa = AMPA_Receptor(n_dendrites * n_outputs, config)
		self.gaba = GABA_Receptor(n_dendrites * n_outputs, config)
		self.integrator = MembraneIntegrator(config['du_dend'], config['dt'])

		self.routing = torch.nn.Parameter(torch.randn(n_inputs) + 2)
		self.surrogate_routing = config['surrogate_spike'] # reuse spiking mechanism as routing mechanism

		self.synapses = NonNegativeLinear(n_inputs, n_dendrites * n_outputs, config)

		self.state = {'u': None}

		self.sum = DendriticSummation(n_dendrites, config)

	def reset(self, batch_size):
		super().reset(batch_size)
		self.nmda.reset(batch_size)
		self.ampa.reset(batch_size)
		self.gaba.reset(batch_size)


	def simulate(self, x):
		gate = self.surrogate_routing(self.routing)
		x_exc = gate * x
		x_inh = (1-gate) * x
		i_exc = self.synapses(x_exc)
		i_inh = self.synapses(x_inh)

		u = self.state['u']

		g_nmda = self.nmda.conductance(i_exc, u)
		g_ampa = self.ampa.conductance(i_exc)
		g_gaba = self.gaba.conductance(i_inh)

		u_new = self.integrator.integrate(u, g_nmda + g_ampa, g_gaba)

		self.state['u'] = u_new

		return u_new

	def forward(self, x):
		return self.sum(self.simulate(x))


class SomaLayer(BiologicalModel):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(n_neurons, config, *args, **kwargs)

		self.lif = LIF(n_neurons, config)
		self.state = {'s': None}

	def reset(self, batch_size):
		self.lif.reset(batch_size)

	def forward(self, x):
		out = self.lif.forward(x)
		self.state['s'] = self.lif.state['s']
		return out



class LinearReadoutLayer(nn.Module):
	def __init__(self, n_inputs, n_outputs, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.li = LI(n_outputs, config)
		self.linear = nn.Linear(n_inputs, n_outputs, config['activate_bias'])
		self.state = {'u': None}

	def reset(self, batch_size):
		self.li.reset(batch_size)


	def forward(self, x):
		out = self.li.forward(
					self.linear(
						x
					)
				)
		self.state['u'] = self.li.state['u']
		return out


class ParallelLayer(nn.Module):
	def __init__(self, layer_list, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.layer_list = layer_list

	def couple(self, x):
		raise NotImplementedError

	def forward(self, x):
		return self.couple(self.layer_list, x)


class MultiplicativeLayer(ParallelLayer):
	def __init__(self, layer_list, *args, **kwargs):
		super().__init__(layer_list, *args, **kwargs)


	def couple(self, x):
		outputs = torch.stack(
			[layer.forward(x) for layer in self.layer_list],
			dim=0
		)
		return torch.prod(outputs, dim=0)

class AdditiveLayer(ParallelLayer):
	def __init__(self, layer_list, *args, **kwargs):
		super().__init__(layer_list, *args, **kwargs)


	def couple(self, x):
		outputs = torch.stack(
			[layer.forward(x) for layer in self.layer_list],
			dim=0
		)
		return torch.sum(outputs, dim=0)


class AffineLayer(ParallelLayer):
	def __init__(self, layer_list, *args, **kwargs):
		super().__init__(layer_list, *args, **kwargs)

		self.weight = nn.Parameter(torch.tensor((1,), device=DEVICE, dtype=DTYPE))
		# self.bias = nn.Parameter(torch.tensor((1,), device=DEVICE, dtype=DTYPE))

	def couple(self, x):
		outputs = torch.stack(
			[layer.forward(x) for layer in self.layer_list],
			dim=0
		)
		return torch.prod(outputs, dim=0) + self.weight * torch.sum(outputs, dim=0)

# d Ax
# d + Ax
# d Ax + Ax + b
# (d + 1) (Ax + b)
# (w * d) (Ax)
# d \in [-1,1]
# LIF without bias

class CoupledDendriticLayer(nn.Module):
	def __init__(self, n_inputs, n_dendrites, n_outputs, CouplingClass, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.dendrites = DendriteLayer(n_inputs, n_dendrites, n_outputs, config)
		self.direct = NonNegativeLinear(n_inputs, n_outputs, config)
		self.coupling_layer = CouplingClass([self.dendrites, self.direct])

	@property
	def state(self):
		return self.dendrites.state


	def reset(self, batch_size):
		self.dendrites.reset(batch_size)


	def forward(self, x):
		return self.coupling_layer.couple(x)


class CompartmentLayer(BiologicalModel):
	def __init__(self, in_features, out_features, config, *args, **kwargs):
		super().__init__(in_features, config, *args, **kwargs)

		self.du_soma = config['du_soma']
		self.du_dend = config['du_dend']

		config_tmp = config
		self.n_dend = config_tmp['n_dendrites']
		self.n_comp = config_tmp['n_compartments']

		self.compartments = nn.ModuleList()
		self.synapses = nn.ModuleList()
		self.relative_conductance = []
		for i in range(self.n_comp):
			g_rel = self.compute_relative_conductance(i)
			self.relative_conductance.append(g_rel)

			config_tmp['du_dend'] = self.du_dend / g_rel
			receptor = NMDA_AMPA_comp(
							out_features,
							config_tmp
							)

			receptor.prepare()
			self.compartments.append(receptor)
			self.synapses.append(
			NonNegativeLinear(
				in_features,
				out_features,
				config,
				),
			)

	def compute_relative_conductance(self, compartment_idx):
		maximum = self.du_dend / self.du_soma
		minimum = 1
		return ((compartment_idx / self.n_comp) * (maximum - minimum)) + (minimum)

	def reset(self, B):
		for comp in self.compartments:
			comp.reset(B)

	def prepare(self):
		for comp in self.compartments:
			comp.prepare()

	def forward(self, x):
		u_0 = 0
		for syn, comp, g_rel in zip(self.synapses, self.compartments, self.relative_conductance):
			u_1 = comp(syn(x) / g_rel, u_0)
			u_0 = u_1
		return u_1


# %%
if __name__ == '__main__':
	from config.ray.pbt import CONFIG

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
	CONFIG['learnable'] = 'all'

	from utils.surrogate import Surrogate
	CONFIG['surrogate_spike'] = Surrogate()

	x = torch.randn((1,700), device=DEVICE, dtype=DTYPE)
	model = CoupledDendriticLayer(700, 3, 64, CONFIG)
	model.to(DEVICE)
	model.reset(1)
	for i in range(99):
		model(x*99)








# %%
