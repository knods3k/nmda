#%%
import os

from ray import tune
import torch

from config.ray.pbt import CONFIG
from ray_training import build_model, build_test_loader
from utils.settings import DEVICE

CONFIG['dt'] = 1e-1

# %%
directory = "/Users/cankayser/Downloads/results/41343"

def retrieve_model(directory):
	e = tune.ExperimentAnalysis(directory)

	best_trial = e.get_best_trial(metric='evaluation', mode='min', scope='all')
	best_checkpoint = e.get_best_checkpoint(best_trial, metric='evaluation', mode='min')
	best_evaluation = best_trial.last_result['evaluation']

	checkpoint = best_checkpoint

	model = build_model(best_trial.config)

	with checkpoint.as_directory() as checkpoint_dir:
		checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location=DEVICE, weights_only=False)

	model.load_state_dict(checkpoint_dict["model_state_dict"])
	return model

# %%
# %%
if __name__ == '__main__':
	model = retrieve_model(directory).to(DEVICE)
	import matplotlib.pyplot as plt
	import numpy as np

	torch.manual_seed(22)
	test_loader = build_test_loader(1, delay_steps=99)
	test_x, test_y = next(iter(test_loader))
	model(test_x.to(DEVICE))

	per_class_logits = model(test_x.to(DEVICE)).to('cpu').detach()[0]
	per_class_probs = torch.nn.functional.softmax(per_class_logits, dim=1)

	T, N = per_class_probs.shape

	one_hot = np.eye(N, dtype=bool)[test_y]

	with plt.style.context('dark'):
		plt.subplots(2,1)
		plt.subplot(211)
		plt.ylabel('Input Neuron')
		plt.xticks([])
		plt.imshow(test_x.detach().cpu()[0].T, origin='lower', cmap='grey_r', interpolation='none')
		plt.colorbar(label='Current [A]')

		plt.subplot(212)
		plt.xlabel('Time [s]')
		plt.ylabel('Probability')
		plt.plot(per_class_probs[:,  one_hot], label='Correct Class')
		plt.plot(per_class_probs[:, ~one_hot], linestyle=':')
		plt.vlines(T-99, 0, 1, 'r', '--', label='Start of Delay Period')
		plt.legend(loc=7)

		plt.tight_layout()
		# plt.savefig('/Users/cankayser/Downloads/activity_NMDA')
		plt.show()


	# %%
