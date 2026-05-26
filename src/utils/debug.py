#%%
import torch

from torch.utils.data import DataLoader
from datasets.spring import Generator
from model import MODEL
from utils.settings import DEVICE, CRITERION

# %%

if __name__ == "__main__":
	from datasets.spring import Generator
	from torch.utils.data import DataLoader

	with torch.autograd.detect_anomaly():


		dataset = Generator()
		loader = DataLoader(dataset, batch_size=32)

		x, y = next(iter(loader))
		loss = CRITERION(MODEL(x.to(DEVICE)), y.to(DEVICE))
		loss.backward()


# %%
