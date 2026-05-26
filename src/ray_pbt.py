#%%
import random

import torch
from torch import nn

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import ExperimentPlateauStopper
from ray.air.integrations.wandb import WandbLoggerCallback

from config.ray.pbt import CONFIG, MAX_HOURS, MAX_BATCH_SIZE, PERTURBATION_INTERVAL, NUM_SAMPLES, ID
from ray_training import train, train_catch_oom
from utils.settings import PROJECT_NAME


FACTORS = [0.1, 0.5, 0.9, 1.1, 2.0, 10.]
WEIGHTS = [0.1, 0.5, 0.9, 0.9, 0.5, 0.1]
RESAMPLE_PROB = .1

def retrieve_perturbation_keys(config):
	integer = []
	floating_point = []
	category = []
	for key in config.keys():
		if type(config[key]) == ray.tune.search.sample.Float:
			floating_point.append(key)
		if type(config[key]) == ray.tune.search.sample.Integer:
			integer.append(key)
		if type(config[key]) == ray.tune.search.sample.Categorical:
			category.append(key)
	return integer, floating_point, category

int_keys, float_keys, cat_keys = retrieve_perturbation_keys(CONFIG)


def explore(config):
	for key in int_keys:
		val = config[key] * random.choices(FACTORS, WEIGHTS)[0]
		config[key] = int(val)
	for key in float_keys:
		val = config[key] * random.choices(FACTORS, WEIGHTS)[0]
		config[key] = float(val)

	if random.random() < RESAMPLE_PROB:
		for key in int_keys + float_keys + cat_keys:
			config[key] = CONFIG[key].sample()

	return config


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
	scheduler = PopulationBasedTraining(
		time_attr="training_iteration",
		perturbation_interval=PERTURBATION_INTERVAL,
		metric="evaluation",
		mode="min",
		resample_probability=0.,
		custom_explore_fn= explore,
	)

	stopper = ExperimentPlateauStopper(
		metric='evaluation',
		mode='min',
		top=NUM_SAMPLES,
		patience=CONFIG['patience']
	)

	tuner = tune.Tuner(
		train,
		tune_config=tune.TuneConfig(
			scheduler = scheduler,
			num_samples = NUM_SAMPLES,
			time_budget_s = MAX_HOURS * 60 * 60,  # total wall-clock time in hours
		),
		run_config=tune.RunConfig(
			name=CONFIG['id'],
			verbose=1,
			# stop=stopper,
        	callbacks=[WandbLoggerCallback(
					project=PROJECT_NAME,
					group=CONFIG['id'],
					log_config=True,
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
	parser.add_argument("-id", "--id", type=str, default=f"{ID}", help="An ID for WandB")
	parser.add_argument("-a", "--architecture", type=str, default='', help="Architecture String")

	try:
		args = parser.parse_args()
	except SystemExit:
		args = parser.parse_args([])
		raise Warning('Could not parse arguments, falling back to default values.')

	CONFIG['id'] = args.id
	CONFIG['architecture'] = args.architecture
	CONFIG['already_initialised'] = []

	tune_with_callback()


# %%
