import torch
import random
import numpy as np

PROJECT_NAME = 'DendriticSNN'

if torch.backends.mps.is_available():
	DEVICE = torch.device('mps')
	DEVICE_FALLBACK = torch.device('cpu')
elif torch.cuda.is_available():
	DEVICE = torch.device('cuda')
	DEVICE_FALLBACK = DEVICE
else:
	DEVICE = torch.device('cpu')
	DEVICE_FALLBACK = DEVICE

MACHINE_EPSILON = torch.finfo(float).eps
BATCH_SIZE = 256
DTYPE = torch.float32

NUM_WORKERS = 0
PREFETCH_FACTOR = None

SHD_PATH = ("/Local/can/datasets/")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
