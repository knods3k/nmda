#%%
import torch
import matplotlib.pyplot as plt

def get_all_parameters(model):
	return torch.cat([p.data.view(-1).cpu() for p in model.parameters() if p.requires_grad])

def get_parameter_histogram(model):
	all_params = get_all_parameters(model)
	hist = torch.histogram(all_params)
	return hist.numpy()

def plot_parameter_histogram(model, bins=None, log=True):
	all_params = get_all_parameters(model)
	plt.figure(figsize=(8, 5))
	plt.hist(all_params, bins=bins, log=log)
	plt.xlabel("Parameter value")
	plt.ylabel("Frequency (log scale)" if log else "Frequency")
	plt.title("Histogram of Trainable Parameters")
	plt.show()

def count_parameters(model):
	return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_n_neurons(model):
	for l in model.layers:
		if type(l).__name__ == 'LIF':
			return l.n_neurons
	return None

# %%
if __name__ == '__main__':
	from datasets.spring import LOADER
	x = (next(iter(LOADER))[0][0]).unsqueeze(0)

	# observe_spikes()

# %%
