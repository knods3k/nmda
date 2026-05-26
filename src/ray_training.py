#%%
import tempfile
import os
import random

import torch
from torch import nn
import numpy as np
from ray import tune
# import wandb

from utils.surrogate import Surrogate
from model import ARCHITECTURES
from datasets.shd_classification import CRITERION, build_loader, build_test_loader
from utils.settings import DEVICE, PROJECT_NAME
from utils.bifurcation_condition import condition
from utils.nmda_init import initialise_nmda_weights
from config.ray.all import ID





def build_surrogate(config):
	return Surrogate(
		sigma=config['sigma'],
		energy=config['energy'],
		type=config['type'],
		)


def build_optimizer(config, model):
	return torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


def build_model(config, surrogate=Surrogate()):
	config['surrogate_spike'] = surrogate
	if config['gam1'] == None:
		config['gam1'] = condition(config['gam0'])
	model = ARCHITECTURES[config['architecture']]
	return model(config)


def evaluate_test_loss(model, test_x, test_y):
		with torch.no_grad():
			model.eval()
			test_loss = CRITERION(model(test_x), test_y)
		return test_loss

def evaluate_accuracy(model, test_x, test_y):
		with torch.no_grad():
			model.eval()
			accuracy = (model(test_x)[:,-1,:].argmax(-1) == test_y).float().mean()
		return -accuracy

def build_evaluation_function(config):
	if config['evaluation'] == 'test_loss':
		return evaluate_test_loss

	if config['evaluation'] == 'accuracy':
		return evaluate_accuracy


def catch_nan(loss):
	if torch.isnan(loss) or torch.isinf(loss):
		tune.report( {'loss':float("inf"), 'evaluation':float("inf")} )
		# raise RuntimeError("NaN detected")


def train_epoch(config, model, loader, optimizer):
	cumulative_loss = 0
	loss = 0
	model.train()
	for i, (x, y) in enumerate(loader):
		if i >= config['steps_per_epoch']:
			break

		model.zero_grad()
		optimizer.zero_grad()

		out = model(x.to(DEVICE))
		loss = CRITERION(out, y.to(DEVICE))
		cumulative_loss += loss.item()

		catch_nan(loss)

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'], error_if_nonfinite=False)
		optimizer.step()

	return cumulative_loss / config['steps_per_epoch']


def train(config=None):
	random.seed(config['seed'])
	np.random.seed(config['seed'])
	torch.manual_seed(config['seed'])

	config['trial_name'] = tune.get_context().get_trial_name()

	# wandb.init(project=PROJECT_NAME, group=config['id'], name=config['trial_name']+'histograms')

	step = 1
	loader = build_loader(config['batch_size'], noise_variance=config['noise_variance'])
	test_loader = build_test_loader(2264)

	surrogate = build_surrogate(config)
	model = build_model(config, surrogate)
	if config['trial_name'] not in config['already_initialised']:
		initialise_nmda_weights(model)
		config['already_initialised'].append(config['trial_name'])
	optimizer = build_optimizer(config, model=model)
	evaluate = build_evaluation_function(config)

	test_x, test_y = next(iter(test_loader))
	test_x = test_x.to(DEVICE) #.unsqueeze(0)
	test_y = test_y.to(DEVICE) #.unsqueeze(0)
	model.to(DEVICE)

	metrics = {
				'evaluation': 0.,
				'loss': 0.,
			}

	checkpoint = tune.get_checkpoint()
	if checkpoint:
		with checkpoint.as_directory() as checkpoint_dir:
			checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), weights_only=False)

		model.load_state_dict(checkpoint_dict["model_state_dict"])
		optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
		for param_group in optimizer.param_groups:
			if "learning_rate" in config:
				param_group["lr"] = config["learning_rate"]
		last_step = checkpoint_dict["step"]
		step = last_step + 1



	for _ in range(config['max_epochs']):
		avg_loss = train_epoch(config, model, loader, optimizer)
		test_loss = evaluate(model, test_x, test_y)

		metrics['loss'] = avg_loss
		metrics['evaluation'] = test_loss.item()

		# for name, param in model.named_parameters():
		# 	try:
		# 		wandb.log({name: wandb.Histogram(param.data.cpu().numpy())})
		# 		wandb.log({name: wandb.Histogram(param.grad.data.cpu().numpy())})
		# 	except Exception as e:
		# 		print("  {} :: data :: {}".format(name, e))

		if step % config["checkpoint_interval"] == 0:
			with tempfile.TemporaryDirectory() as tmpdir:
				torch.save(
					{
						"step": step,
						"model_state_dict": model.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"config": config
					},
					os.path.join(tmpdir, "checkpoint.pt"),
				)
				tune.report(metrics, checkpoint=tune.Checkpoint.from_directory(tmpdir))
		else:
			tune.report(metrics)

		step += 1

def train_catch_oom(config=None):
	try:
		train(config)
	except MemoryError:
			tune.report( {'loss':float("inf"), 'evaluation':float("inf")} )


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
	CONFIG['learnable'] = 'none'
	CONFIG['noise_variance'] = 0
	CONFIG['architecture'] = 'NMDA_AMPA_LIF_SNN'

	# with torch.autograd.detect_anomaly():
	# 	train(CONFIG)


	config = CONFIG
	loader = build_loader(config['batch_size'], noise_variance=config['noise_variance'])
	test_loader = build_test_loader(2264)

	surrogate = build_surrogate(config)
	model = build_model(config, surrogate)
	initialise_nmda_weights(model)
	optimizer = build_optimizer(config, model=model)
	evaluate = build_evaluation_function(config)

	from utils.diagnosis import count_trainable_parameters
	count_trainable_parameters(model)
	x,y = next(iter(loader))
	x = x.to(DEVICE)
	model.to(DEVICE)
	model(x)

	train(config)

# %%
