#%%
import numpy as np
from scipy.special import lambertw as W
from scipy.special import expit as sigmoid
from neurons import DendriteLayer
import torch
# from ray_training import build_loader
from utils.settings import DTYPE, DEVICE

# %%
AVG_RATE = .1
MAX_LENGTH = 100

def compute_spike_statistics(data_loader):
	x,y = next(iter(data_loader))
	avg_rate = x.mean(dim=(0,1))
	MAX_LENGTH = x.shape[1]
	return avg_rate, MAX_LENGTH


#%%
def h(u, gam0=.5, gam1=8.):
	return sigmoid(gam1 * (u - gam0))

def initialise_nmda_weights(model, avg_rate=AVG_RATE, t=MAX_LENGTH, seed=None):
	for layer in model.layers:
		# avg_rate *= 0.02
		if isinstance(layer, DendriteLayer):
			c0 = layer.nmda.gam0
			c1 = layer.nmda.gam1
			umin = (1/2) - np.sqrt((1/4) - (1/c1))
			umax = (1/c1) * np.log((c1/4) - 1) + c0
			rho_u = layer.nmda.du_dend_log.exp().detach().cpu().mean()
			gmin = rho_u * (umin / h(umin, c0, c1)*(1-umin))
			gmax = rho_u * (umax / h(umax, c0, c1)*(1-umax))

			rho_d = layer.nmda.dv_log.exp().detach().cpu().mean()
			rho_r = layer.nmda.dw_log.exp().detach().cpu().mean()

			time_weighting = ((1- np.exp(-rho_d*t)) / rho_d) - ((1- np.exp(-rho_r*t)) / rho_r)
			A = construct_nonnegative_matrix(avg_rate, layer.synapses.in_features, layer.synapses.out_features, gmin*time_weighting, gmax*time_weighting)
			# A += (5e-1*torch.randn(A.shape, ))**2

			layer.synapses.weight = torch.nn.Parameter(A.T)




def construct_nonnegative_matrix(avg_rate, n_inputs, n_outputs, g_min, g_max, seed=None):
	if seed is not None:
		torch.manual_seed(seed)

	avg_rate = torch.tensor(avg_rate)

	if avg_rate.ndim == 0:
		avg_rate = torch.ones(n_inputs, device=DEVICE, dtype=DTYPE) * avg_rate
	else:
		avg_rate = avg_rate.flatten()
		assert avg_rate.shape[0] == n_inputs

	sigma = g_max - g_min

	z = torch.randn(n_outputs, device=DEVICE, dtype=DTYPE)

	g_target = 	g_max + (sigma * z)
	A = torch.randn((n_inputs, n_outputs), device=DEVICE, dtype=DTYPE).abs()
	current = (A * avg_rate[:,None]).sum(dim=0)
	eps = 1e-5
	scale = g_target / (current + eps)
	A = A * scale

	return A


# %%
if __name__ == "__main__":
	from ray_training import build_loader, build_model
	loader = build_loader(256)
	avg_rate, _ = compute_spike_statistics(loader)
	A = construct_nonnegative_matrix(.1, 700, 32, 3, 6, 1234)
	# A += (1e-3 * torch.randn(A.shape, ))**2

	# model = build_model()


# %%
