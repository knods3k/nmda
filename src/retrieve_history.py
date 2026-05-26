#%%
import numpy as np
from ray import tune
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
directory = '/Users/cankayser/Downloads/results/43230'

def get_array(list):
	max_len = max([len(arr) for arr in list])

	arr = np.full((len(list), max_len), np.nan)
	for i in range(len(list)):
		for j in range(len(list[i])):
			arr[i,j] = list[i][j]

	return arr

def retrieve_history(directory):
	e = tune.ExperimentAnalysis(directory)

	dfs = list(e.trial_dataframes.values())

	acc_list = [df['evaluation'].to_numpy() for df in dfs]
	t_list = [df['timestamp'].to_numpy() for df in dfs]


	acc = get_array(acc_list) * (-100)
	acc[acc == -np.inf] = np.nan
	t = get_array(t_list)
	t = ((t - np.nanmin(t)) / (60)) / 60
	return t, acc

def retrieve_hyperparams(directory):
	e = tune.ExperimentAnalysis(directory)

	dfs = list(e.trial_dataframes.values())

	noise_list = [df['config/noise_variance'].to_numpy() for df in dfs]
	sigma_list = [df['config/sigma'].to_numpy() for df in dfs]
	energy_list = [df['config/energy'].to_numpy() for df in dfs]
	learning_rate_list = [df['config/learning_rate'].to_numpy() for df in dfs]
	t_list = [df['timestamp'].to_numpy() for df in dfs]

	hyperparams = {
		"Noise": get_array(noise_list),
		"Learning Rate": get_array(learning_rate_list),
		"Surrogate Magnitude": get_array(energy_list),
		"Surrogate Spread": get_array(sigma_list),
	}
	t = get_array(t_list)
	t = ((t - np.nanmin(t)) / (60)) / 60
	return t, hyperparams


if __name__ == '__main__':
	t, acc = retrieve_history(directory)
	t, hyperparams = retrieve_hyperparams(directory)
	symbols = {
		"Noise": r"$\lambda$",
		"Surrogate Magnitude": r"$\beta_0$",
		"Surrogate Spread": r"$\beta_0$",
		"Learning Rate": r"$\tau$",
		}
	# %%
	with plt.style.context('save'):
		plt.plot(acc.T, c='C0', linestyle='-', alpha=.3)
		plt.plot(np.nanmean(acc, axis=0), alpha=1., label=r'Average')
		plt.plot(np.nanmax(acc, axis=0), label=r'Maximum')
		plt.ylim(0,100)
		plt.grid(linestyle=':')
		plt.ylabel(r'Test Accuracy [$\%$]')
		plt.xlabel(r'Step')
		plt.legend(loc='lower right')
		# plt.title(fr'Maximum Accuracy: {np.nanmax(acc): 2.0f} $\%$')
		plt.savefig('../thesis/figures/history')
		plt.show()


		plt.subplots(2,2, sharex=True, layout='compressed')
		for i, (n,p) in enumerate(hyperparams.items()):
			plt.subplot(2,2,(i+1))
			plt.plot(p.T, linestyle='-', c='C1', alpha=.66)
			# plt.plot(np.nanmean(acc, axis=0), alpha=1., label='Average')
			# plt.plot(np.nanmax(acc, axis=0), label='Maximum')
			# plt.ylim(0,100)
			# plt.grid(linestyle=':')
			plt.ylabel(rf'{n} {symbols[n]}')
			plt.yscale('log')
			# plt.gca().yaxis.set_major_formatter(
			# 	mpl.ticker.FuncFormatter(lambda y, _: f"{int(np.log10(y))}")
			# )
			if i > 1:
				plt.xlabel('Step')
			if i % 2 == 1:
				plt.gca().yaxis.set_label_position("right")
				plt.gca().yaxis.tick_right()
		plt.savefig('../thesis/figures/history_hyperparams')
		plt.show()



	# %%
