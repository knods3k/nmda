#%%
import os

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets.delayed_shd_ssc import SpikingDataset

from utils.settings import NUM_WORKERS, PREFETCH_FACTOR

CE = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
def sequence_cross_entropy(logits, targets):
    B, T, N_CLASSES = logits.shape
    logits_flat = logits.reshape(-1, N_CLASSES)
    targets_flat = targets.unsqueeze(-1).expand(B, T).reshape(-1)
    return CE(logits_flat, targets_flat)

CRITERION = sequence_cross_entropy


from utils.settings import SHD_PATH
if os.path.exists(SHD_PATH):
	DIRECTORY_NAME = os.path.dirname(SHD_PATH)
else:
	DIRECTORY_NAME = os.path.dirname(__file__)
DEFAULT_DATA_PATH = os.path.join(DIRECTORY_NAME, "shd_ssc_data")

class Generator(IterableDataset):
	def __init__(self, dataset: SpikingDataset, shuffle=True, **kwargs):
		self.dataset = dataset
		self.shuffle = shuffle
		self.N_samples = len(self.dataset)
		self.N = dataset.nb_units

	def generateBatch(self, batch):
		xs, ys, mask = zip(*batch)
		xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
		ys = torch.LongTensor(ys)
		return xs, ys

	def __iter__(self):
		if self.shuffle:
			while True:
				idx = torch.randint(0, self.N_samples, ()).item()
				yield self.dataset[idx]
		else:
			while True:
				for idx in range(self.N_samples):
					yield self.dataset[idx]

#%%
def build(batch_size, split='train', shuffle=True, **kwargs):
    base = SpikingDataset(
        dataset_name="shd",
        data_folder=DEFAULT_DATA_PATH,
        split=split,
        nb_steps=140,
        # delay_steps=20,
		**kwargs,
    )

    iterable = Generator(base, shuffle=shuffle)

    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=iterable.generateBatch,
		num_workers=NUM_WORKERS,
		pin_memory=True,
		persistent_workers=NUM_WORKERS>0,
		prefetch_factor=PREFETCH_FACTOR,

    )

    return loader

def build_loader(batch_size=1, **kwargs):
	return build(batch_size, split='train', **kwargs)

def build_test_loader(batch_size=1, **kwargs):
	loader = build(batch_size, split='test', shuffle=False, **kwargs)
	return loader

if __name__ == '__main__':
	loader = build_loader(3)
	for i, (x, y) in enumerate(loader):
		print(x)
		break
# %%
