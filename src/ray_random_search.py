#%%
import torch
from torch import nn

import ray
from ray import tune
from ray.tune.search import BasicVariantGenerator
from ray.air.integrations.wandb import WandbLoggerCallback

from config.ray.pbt import CONFIG
from config.ray.utils import PatienceStopper
from ray_training import train, train_catch_oom
from utils.settings import PROJECT_NAME
from config.ray.all import ID

MAX_HOURS = 70


train = tune.with_resources(
	train_catch_oom,
	{
		"cpu": 4,
		"gpu": (1/2 if torch.cuda.is_available() else 0),
		"memory": 12 * 1024**3
	}
)


def tune_with_callback():
	ray.init()
	tuner = tune.Tuner(
		train,
		tune_config=tune.TuneConfig(
        	search_alg=BasicVariantGenerator(),
			metric="evaluation",
			mode="min",
			num_samples=-1,
			time_budget_s = MAX_HOURS * 60 * 60,  # total wall-clock time in hours
		),
		run_config=tune.RunConfig(
			name=CONFIG['id'],
			verbose=1,
			# stop=stopper,
        	# callbacks=[WandbLoggerCallback(
			# 		project=PROJECT_NAME,
			# 		group=CONFIG['id'],
			# 		log_config=True,
			# 	)]
		),
		param_space=CONFIG,
	)

	return tuner.fit()


#%%
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("-id", "--id", type=str, default=f"{ID}", help="An ID for WandB")
	parser.add_argument("-a", "--architecture", type=str, default='', help="Architecture String")

	try:
		args = parser.parse_args()
	except SystemExit:
		args = parser.parse_args([])
		# raise Warning('Could not parse arguments, falling back to default values.')

	CONFIG['id'] = args.id
	CONFIG['architecture'] = args.architecture
	CONFIG['already_initialised'] = []

	with torch.autograd.detect_anomaly(check_nan=True):
		tune_with_callback()

# %%
