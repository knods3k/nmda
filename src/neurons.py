#%%
import torch
import torch.nn as nn

from utils.settings import DEVICE, MACHINE_EPSILON, DTYPE


class NeuronModel(nn.Module):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.n_neurons = n_neurons
		self.state = {}

		self._constructor_args = {
			'n_neurons': n_neurons,
			'args': args,
			'kwargs': kwargs.copy()
		}

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

	def clone_with_new_dim(self, new_dim):
		args = self._constructor_args['args']
		kwargs = self._constructor_args['kwargs'].copy()
		return self.__class__(new_dim, *args, **kwargs)

	def reset(self, batch_size):
		for key in self.state.keys():
			self.state[key] = torch.zeros((batch_size, self.n_neurons), device=DEVICE, dtype=DTYPE)
		return self.state


class DendriteModel(nn.Module):
	def __init__(self, n_dendrites, config, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.n_dendrites = n_dendrites

		dt = torch.exp(torch.tensor(config['dt'], device=DEVICE, dtype=DTYPE))
		du = torch.exp(torch.tensor(config['du_dend'], device=DEVICE, dtype=DTYPE))

		if 'dendritic_scaling' in config:
			winit = config['dendritic_scaling']
			self.w = nn.Parameter(torch.tensor(winit), requires_grad=False)
		else:
			winit = 1.
			self.w = nn.Parameter(torch.tensor(winit), requires_grad=True)

	def forward(self, x):
		B, D = x.shape
		return (x.view((B, -1, self.n_dendrites)) * torch.exp(self.w) ).mean(-1)



class NonNegativeLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight= nn.Parameter(torch.randn(out_features, in_features, dtype=DTYPE, device=DEVICE))
		self.bias= nn.Parameter(torch.randn(out_features, dtype=DTYPE, device=DEVICE))

	def forward(self, input):
		return nn.functional.linear(input, self.weight.abs(), self.bias.abs())


class LI(NeuronModel):
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

	def prepare(self):
		self.dt = torch.exp(self.dt_log)
		self.du = torch.exp(self.du_soma_log)
		self.du_inv = 1/self.du
		self.du_exp = torch.exp(-self.du*self.dt)
		self.du_exp_m1 = -torch.expm1(-self.du*self.dt)

	def forward(self, x):

		self.state['i'] = x

		u = self.state['u']
		i = self.state['i']

		u_new = (self.du_exp * u) + (self.du_exp_m1 * i * self.du_inv)

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

		u_hat = (self.du_exp * u) + (self.du_exp_m1 * i * self.du_inv)
		u_new = u_hat * (1 - s)

		self.state['u'] = u_new
		self.state['s'] = s
		return s


class NMDA(LI):
	def __init__(self, n_inputs, n_outputs, config, *args, **kwargs):
		super().__init__(n_outputs, config, *args, **kwargs)

		self.state = {'i': None, 'u': None, 'w': None,  'v': None}

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

		self.exc_linear = NonNegativeLinear(n_inputs, n_outputs)

	# def reset(self, batch_size):
	# 	u1 = 1 - scipy.special.lambertw(-np.exp(1) / self.gamma, k=-1).real
	# 	u2 = 1 - scipy.special.lambertw(-np.exp(1) / self.gamma, k=0).real
	# 	u1 = torch.tensor(u1, device=DEVICE, dtype=DTYPE)
	# 	u2 = torch.tensor(u2, device=DEVICE, dtype=DTYPE)
	# 	j1 = self.du * (u1 / self.h(u1))
	# 	j2 = self.du * (u2 / self.h(u2))
	# 	self.state['u'] = torch.zeros((batch_size, self.n_neurons), device=DEVICE, dtype=DTYPE) + ((u1-u2)/2)
	# 	self.state['v'] = torch.zeros((batch_size, self.n_neurons), device=DEVICE, dtype=DTYPE) + ((j1-j2)/2)
	# 	self.state['w'] = torch.zeros((batch_size, self.n_neurons), device=DEVICE, dtype=DTYPE) + 0
	# 	return self.state

	@staticmethod
	def h(u, gam0=.5, gam1=8.):
		return torch.sigmoid(gam1 * (u - gam0))


	def prepare(self):
		super().prepare()
		self.du = torch.exp(self.du_dend_log)
		self.du_inv = 1/self.du
		self.du_exp = torch.exp(-self.du*self.dt)
		self.du_exp_m1 = -torch.expm1(-self.du*self.dt)

		self.dv = torch.exp(self.dv_log)
		self.dv_inv = 1/self.dv
		self.dv_exp = torch.exp(-self.dv*self.dt)
		self.dv_exp_m1 = -torch.expm1(-self.dv*self.dt)

		self.dw = torch.exp(self.dw_log)
		self.dw_inv = 1/self.dw
		self.dw_exp = torch.exp(-self.dw*self.dt)
		self.dw_exp_m1 = -torch.expm1(-self.dw*self.dt)

		self.norm = (self.dv*self.dw) / (self.dw - self.dv)
		# self.norm = -torch.exp(-self.dw*torch.log(self.dv/self.dw)/(self.dv - self.dw)) + torch.exp(-self.dv*torch.log(self.dv/self.dw)/(self.dv - self.dw))


	def forward(self, x):
		i_exc = self.exc_linear(x)
		u = self.state['u']
		w = self.state['w']
		v = self.state['v']

		v_new = (self.dv_exp * v) + (self.dv_exp_m1 * i_exc * self.dv_inv)
		w_new = (self.dw_exp * w) + (self.dw_exp_m1 * i_exc * self.dw_inv)

		g_nmda = self.norm * (v_new - w_new) * self.h(u, gam0=self.gam0, gam1=self.gam1)
		u_new = (u + self.dt * g_nmda) / (1 + self.dt * (self.du + g_nmda))

		self.state['u'] = u_new
		self.state['v'] = v_new
		self.state['w'] = w_new

		return u_new


class NMDA_AMPA(NMDA):
	def __init__(self, n_inputs, n_outputs, config, *args, **kwargs):
		super().__init__(n_inputs, n_outputs, config, *args, **kwargs)

		self.register_logarithmic_parameters({'relative_concentration_log': config['relative_concentration']}, learnable='all')
		self.register_logarithmic_parameters({'dg_log': config['dg']}, learnable=config['learnable'])

		self.state['g_ampa'] = None


	def prepare(self):
		super().prepare()
		self.lam = torch.exp(self.relative_concentration_log)

		self.dg = torch.exp(self.dg_log)
		self.dg_inv = 1/self.dg
		self.dg_exp = torch.exp(-self.dg*self.dt)
		self.dg_exp_m1 = -torch.expm1(-self.dg*self.dt)


	def forward(self, x):
		i_exc = self.exc_linear(x)
		i = self.state['i']
		u = self.state['u']
		w = self.state['w']
		v = self.state['v']
		g_ampa = self.state['g_ampa']

		v_new = (self.dv_exp * v) + (self.dv_exp_m1 * i_exc * self.dv_inv)
		w_new = (self.dw_exp * w) + (self.dw_exp_m1 * i_exc * self.dw_inv)
		g_ampa_new = (self.dg_exp * g_ampa) + (self.dg_exp_m1 * i_exc * self.dg_inv * self.lam)
		g_nmda_new = self.norm * (v_new - w_new) * self.h(u, gam0=self.gam0, gam1=self.gam1)

		g_total = g_nmda_new + g_ampa_new
		u_new = (u + self.dt * g_total) / (1 + self.dt * (self.du + g_total))

		self.state['u'] = u_new
		self.state['v'] = v_new
		self.state['w'] = w_new
		self.state['g_ampa'] = g_ampa_new

		return u_new



class NMDA_AMPA_comp(NMDA_AMPA):
	def __init__(self, n_neurons, config, *args, **kwargs):
		super().__init__(n_neurons, config, *args, **kwargs)

		self.register_logarithmic_parameters({'g_coupling_log': config['coupling_conductance']}, learnable='all')

	def prepare(self):
		super().prepare()
		self.g_coupling = torch.sigmoid((self.g_coupling_log))


	def forward(self, x, ud=0):

		self.state['i'] = x
		i = self.state['i']
		u = self.state['u']
		w = self.state['w']
		v = self.state['v']
		g = self.state['g']

		v_new = (self.dv_exp * v) + (self.dv_exp_m1 * i * self.dv_inv)
		w_new = (self.dw_exp * w) + (self.dw_exp_m1 * i * self.dw_inv)
		g_new = (self.dg_exp * g) + (self.dw_exp_m1 * i * self.dw_inv * self.lam)

		drive = (self.norm * (v_new - w_new) * self.h(u, gam0=self.gam0, gam1=self.gam1)) + g
		coupling = (self.g_coupling * ud)
		u_new = (u + self.dt * (drive + coupling)) / (1 + self.dt * (self.du + drive + self.g_coupling))

		self.state['u'] = u_new
		self.state['v'] = v_new
		self.state['w'] = w_new
		self.state['g'] = g_new

		return u_new

class NMDA_AMPA_GABA(NMDA_AMPA):
	def __init__(self, n_inputs, n_outputs, config, *args, **kwargs):
		super().__init__(n_inputs, n_outputs, config, *args, **kwargs)

		self.state['g_gaba'] = None

		self.routing = torch.nn.Parameter(torch.randn(n_inputs)) # start biased toward excitation to ensure early spiking
		self.surrogate_routing = config['surrogate_spike'] # reuse surrogate spiking mechanism to route signals to excitation/inhibition

	def prepare(self):
		super().prepare()
		self.norm = -torch.exp(-self.dw*torch.log(self.dv/self.dw)/(self.dv - self.dw)) + torch.exp(-self.dv*torch.log(self.dv/self.dw)/(self.dv - self.dw))



	def forward(self, x):
		gate = self.surrogate_routing(self.routing)
		x_exc = gate * x # signal gets cut in half -> double the strength
		x_inh = (1-gate) * x
		i_exc = self.exc_linear(x_exc)
		i_inh = self.exc_linear(x_inh)
		u = self.state['u']
		w = self.state['w']
		v = self.state['v']
		g_ampa = self.state['g_ampa']
		g_gaba = self.state['g_gaba']

		lam = 1 + (torch.sigmoid(self.lam)*3)
		v_new = (self.dv_exp * v) + (self.dv_exp_m1 * i_exc * self.dv_inv)
		w_new = (self.dw_exp * w) + (self.dw_exp_m1 * i_exc * self.dw_inv)
		g_ampa_new = (self.dg_exp * g_ampa) + (self.dg_exp_m1 * i_exc * self.dg_inv * lam)
		g_gaba_new = (self.dg_exp * g_gaba) + (self.dg_exp_m1 * i_inh * self.dg_inv)
		g_nmda_new = self.norm * (v_new - w_new) * self.h(u, gam0=self.gam0, gam1=self.gam1)

		u_new = (u + self.dt * (g_nmda_new + g_ampa_new - g_gaba_new)) / (1 + self.dt * (self.du + g_nmda_new + g_ampa_new + g_gaba_new))

		self.state['u'] = u_new
		self.state['v'] = v_new
		self.state['w'] = w_new
		self.state['g_ampa'] = g_ampa_new
		self.state['g_gaba'] = g_gaba_new

		return u_new




class NMDA_AMPA_ablation(NMDA_AMPA):
	@staticmethod
	def h(u, gam0=.5, gam1=8.):
		return 1.


class COMPARTMENTS(NeuronModel):
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
				out_features
				),
			)

	def compute_relative_conductance(self, comp_idx):
		maximum = self.du_dend / self.du_soma
		minimum = 1
		return ((comp_idx / self.n_comp) * (maximum - minimum)) + (minimum)

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

	x = torch.randn((1,21,700), device=DEVICE, dtype=DTYPE)
	model = COMPARTMENTS(700, 64, CONFIG)
	model.to(DEVICE)
	model(x)








# %%
