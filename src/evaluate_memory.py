#%%
import torch
import matplotlib.pyplot as plt

from utils.diagnosis import get_n_dendrites, get_n_neurons
from utils.settings import DEVICE, MACHINE_EPSILON
from retrieve_model import retrieve_model
from datasets.shd_classification import build_test_loader

#%%
# %%
test_loader = build_test_loader()

test_x, test_y = next(iter(test_loader))
test_x = test_x.to(DEVICE)
test_y = test_y.to(DEVICE)

def lagged_mutual_information(spikes, max_lag=None):
	s = spikes.float()

	B, T, N = s.shape

	if max_lag is None:
		max_lag = T
	else:
		max_lag = min(T, max_lag)

	results_self = []
	results_cross = []
	results_all = []

	for lag in range(max_lag):
		x = s[:,:T-lag,:]
		y = s[:,lag:,:]

		x = x.T.reshape(N, -1)
		y = y.T.reshape(N, -1)

		px1 = x.mean(dim=1)
		py1 = y.mean(dim=1)

		px0 = 1 - px1
		py0 = 1 - py1

		p11 = x @ y.T / x.shape[1]
		p10 = px1[:,None] - p11
		p01 = py1[:,None] - p11
		p00 = 1 - p11 - p10 - p01

		px = torch.stack([px0, px1], dim=1)
		py = torch.stack([py0, py1], dim=1)

		p = torch.stack(
			[
				torch.stack([p00, p01], dim=-1),
				torch.stack([p10, p11], dim=-1),
			],
			dim=-2
		)

		expected = px[:,None, :, None] * py[None,:,None,:]

		mutual_information = (
			p.clamp_min(MACHINE_EPSILON)
			* torch.log(p.clamp_min(MACHINE_EPSILON)
			   / expected.clamp_min(MACHINE_EPSILON))
		).sum(dim=(-1,-2))


		diagonal = torch.diagonal(mutual_information)
		mi_self = diagonal.mean()

		mi_cross = (
			mutual_information.sum() - diagonal.sum()
		) / (N * (N - 1))

		mi_all = mutual_information.mean()

		results_self.append(mi_self)
		results_cross.append(mi_cross)
		results_all.append(mi_all)

	results_self = torch.stack(results_self)
	results_cross = torch.stack(results_cross)
	results_all = torch.stack(results_all)

	results_self = results_self / results_self[0]
	results_cross = results_cross / results_cross[0]
	results_all = results_all / results_all[0]

	lags = torch.arange(max_lag, device=DEVICE)
	return lags, results_self, results_cross, results_all



dir = '/Users/cankayser/Downloads/ray_results/59237'

for dir in [
	'/Users/cankayser/Downloads/ray_results/63699',
	'/Users/cankayser/Downloads/ray_results/63360',
	'/Users/cankayser/Downloads/ray_results/59237',
]:
	model = retrieve_model(dir).to(DEVICE)

	states = model.test(test_x)

	s = list(states[1].values())[0]
	# print(s.mean())


	lag, mi_self, mi_cross, mi_all = lagged_mutual_information(s)

	n = get_n_neurons(model)
	d = get_n_dendrites(model)

	name_str = type(model).__name__ + f"_{n}_{d}"
	plt.plot(lag.cpu(), mi_cross.cpu(), label=name_str)
plt.legend()
plt.show()


# %%
