from ray import tune
from datetime import date

from config.ray.all import TODAY, ID, MAX_HOURS, MAX_BATCH_SIZE, NUM_SAMPLES

PERTURBATION_INTERVAL = 8

CONFIG = {
	"architecture": '',
	# "learnable": tune.choice(['none', 'one', 'all']),
	"learnable": 'none',
	"n_inputs": 700,
	"n_hidden": 128,
	"n_dendrites": 16,
	"n_compartments": 2,
	"n_outputs": 20,
	"threshold": 1.,
	"du_dend": (1/2),
	"du_soma": (1/10),
	"dv": (1/50),
	"dw": (1/10),
	"dg": (1/3),
	# "dt": tune.loguniform(1e-2,1),
	"dt": 1e-2,
	"gam0": 0.75,
	"gam1": 6.66,
	"dendritic_scaling": 3.9,
	"relative_concentration": 1.,
	"coupling_conductance": 1.,
	"sigma": tune.loguniform(.001, 1.),
	"energy": tune.loguniform(.001, 1.),
	"type": tune.choice(['gauss', 'laplace', 'cauchy']),
	"batch_size": tune.randint(128,1024),
	"learning_rate": tune.loguniform(1e-5, 1),
	"max_epochs": 99999,
	"steps_per_epoch": 128,
	"checkpoint_interval": 8,
	"clip": 1.,
	"patience": 128,
	"evaluation": 'accuracy',
	"noise_variance": tune.loguniform(1e-5,1.),
	"seed": tune.randint(0, 9999),
	"id": ID,
	"already_initialised": [],
	}
