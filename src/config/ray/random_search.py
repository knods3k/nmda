from ray import tune

CONFIG = {
	"sigma": tune.loguniform(.1, 10.),
	"energy": tune.loguniform(.1, 10.),
	"type": tune.choice(['gauss', 'laplace', 'cauchy']),
	"learnable": 'none',
	"n_hidden": 512,
	"du": 2.,
	"dv": 1.,
	"dw": 8.,
	"dt": 0.1,
	"gam0": 8.,
	"gam1": .5,
	"batch_size": 512,
	"learning_rate": tune.loguniform(1e-5, 1e-1),
	"max_epochs": 99,
	"steps_per_epoch": 128,
	"checkpoint_interval": 4,
	"clip": 1.,
	"patience": 8,
	"evaluation": 'accuracy',
	}
