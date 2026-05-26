#%%
import torch
from torch import nn

import ray
from ray import tune
from ray.tune.search import BasicVariantGenerator
from ray.air.integrations.wandb import WandbLoggerCallback

from config.ray.pbt import CONFIG
from config.ray.utils import PatienceStopper
from ray_training import train
from utils.settings import PROJECT_NAME

MAX_HOURS = 70

train = tune.with_resources(train, {"cpu": 1, "gpu": 1})
def tune_with_callback():
	ray.init()

	tuner = tune.Tuner(
		train,
		tune_config=tune.TuneConfig(
        	search_alg=BasicVariantGenerator(),
			metric='test_loss',
			mode='min',
			num_samples=-1,
			time_budget_s=MAX_HOURS * 60 * 60,  # total wall-clock time in hours
		),
		run_config=tune.RunConfig(
			keep_checkpoints_num=1,
			checkpoint_score_attr="test_loss",
			verbose=1,
			stop=PatienceStopper(
					metric="test_loss",
					patience=CONFIG['patience'],
				),
        	callbacks=[WandbLoggerCallback(
					project=PROJECT_NAME,
					group=CONFIG['id'],
				)]
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

	parser.add_argument("-st", "--surrogate_type", type=str, default='default', help="The type of surrogate gradient to use")
	parser.add_argument("-id", "--id", type=str, default='default', help="An ID for WandB")

	try:
		args = parser.parse_args()
	except SystemExit:
		args = parser.parse_args([])
		raise Warning('Could not parse arguments, falling back to default values.')

	surrogate_type = args.surrogate_type
	ID = args.id

	if ID == 'default':
		ID = surrogate_type


	CONFIG['id'] = ID

	if surrogate_type == 'default':
		CONFIG['type'] = tune.grid_search(['gauss', 'laplace', 'cauchy'])
	else:
		CONFIG['type'] = surrogate_type


	result_grid = tune_with_callback()
	df = result_grid.get_dataframe()
	df.to_csv(f'surrogate_search.csv')


# %%
