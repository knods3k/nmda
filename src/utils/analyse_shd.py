#%%
import numpy as np
from scipy.special import lambertw as W
from model import NonNegativeLinear, NMDA
import torch

# %%
def compute_spike_statistics(data_loader):
	avg_rate = 0.
	mse = 0.
	n = 0
	for i in range(999):
		test_x, test_y = next(iter(data_loader))
		x = test_x.mean(dim=1)

		diff = x - avg_rate
		n += 1
		avg_rate += ((diff) / n)
		mse += diff * (x - avg_rate)
	avg_rate = avg_rate[0]
	avg_rate_var = (mse / (n-1))[0]
	return avg_rate, avg_rate_var

# %%

def initialise_nmda_weights(model, data_loader):
	avg_rate, avg_rate_var = compute_spike_statistics(data_loader)
	for layer in model.layers:
		if isinstance(layer, NMDA):
			c0 = layer.gam0.exp().detach().cpu().numpy()
			c1 = layer.gam1.exp().detach().cpu().numpy()
			umin = ((-1/c1) * W((c1/2) * np.exp(c1*(.5-c0))).real) + .5
			umin = float(umin)
			umin = torch.tensor(umin)
			c1 = torch.tensor(c1)
			c0 = torch.tensor(c0)
			rho = layer.du.exp().detach().cpu()
			gmin = (rho / layer.h(umin,c0,c1)) * (umin / (1-umin))

	for layer in model.layers:
		if isinstance(layer, NonNegativeLinear):
			N, M = layer.weight.shape
			mean_exp = (avg_rate.view(1, M).expand(N, M)) * M
			std_exp = torch.sqrt(avg_rate_var).view(1,M).expand(N,M)

			mean = (gmin) / mean_exp
			std = std_exp
			var = std**2
			sigma2 = torch.log(1 + var / mean**2)
			sigma = torch.sqrt(sigma2)
			mu = torch.log(mean) - 0.5 * sigma2
			layer.weight = torch.nn.Parameter(torch.normal(mean=mu, std=sigma))
			layer.bias = torch.nn.Parameter(-torch.ones(N)*999)
			break







# %%
