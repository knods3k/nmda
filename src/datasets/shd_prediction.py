#%%
import os

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets.delayed_shd_ssc import SpikingDataset

CRITERION = torch.nn.MSELoss(reduction='mean')


DIRECTORY_NAME = os.path.dirname(__file__)
DEFAULT_DATA_PATH = os.path.join(DIRECTORY_NAME, "shd_ssc_data")

class Generator(IterableDataset):
	def __init__(self, dataset: SpikingDataset, shuffle=True, **kwargs):
		self.dataset = dataset
		self.shuffle = shuffle
		self.N_samples = len(self.dataset)
		self.N = dataset.nb_units

	def generateBatch(self, batch):
		xs, ys, _ = zip(*batch)
		xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
		ys = torch.LongTensor(ys)
		return xs[:,:-1], xs[:,1:]

	def __iter__(self):
		if self.shuffle:
			while True:
				idx = torch.randint(0, self.N, ()).item()
				yield self.dataset[idx]
		else:
			while True:
				for idx in range(self.N):
					yield self.dataset[idx]

#%%
def build_loader(batch_size):
    base = SpikingDataset(
        dataset_name="shd",
        data_folder=DEFAULT_DATA_PATH,
        split="train",
        nb_steps=140,
        delay_steps=20,
    )

    iterable = Generator(base)

    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=iterable.generateBatch
    )

    return loader

if __name__ == '__main__':
	loader = build_loader(3)
	for i, (x, y) in enumerate(loader):
		print(x)
		break
