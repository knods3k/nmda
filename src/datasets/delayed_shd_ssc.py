#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
# Modified by Younes Bouhadjar
#
"""
This is where the dataloader is defined for the SHD and SSC datasets.
This implementation includes support for delayed classification.
"""
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpikingDataset(Dataset):
	"""
	Dataset class for the Spiking Heidelberg Digits (SHD) or
	Spiking Speech Commands (SSC) dataset.

	This class implements delayed classification.

	Arguments
	---------
	dataset_name : str
		Name of the dataset, either shd or ssc.
	data_folder : str
		Path to folder containing the dataset (h5py file).
	split : str
		Split of the SHD dataset, must be either "train" or "test".
	nb_steps : int
		Number of time steps for the generated spike trains.
	delay_steps : int
		Number of zero-padded timesteps to add after the sequence for delayed classification.
		If 0, no padding is added but classification mask still points to the last timestep.
	noise_variance : float
		Variance of additive noise.
		If 0, no noise is added.
	"""

	def __init__(
		self,
		dataset_name,
		data_folder,
		split,
		nb_steps=140,
		max_time=1.4,
		num_bins=1,
		delay_steps=0,
		noise_variance=0
	):
		self.nb_steps = nb_steps
		self.num_bins = num_bins
		self.nb_units = 700
		self.nb_units_binned = self.nb_units//self.num_bins
		self.max_time = max_time
		self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)
		self.delay_steps = delay_steps
		self.noise_variance = noise_variance

		self.filename = f"{data_folder}/{dataset_name}_{split}.h5"
		self.file = None
		self.firing_times = None
		self.units_fired = None
		self.labels = None

		with h5py.File(self.filename, 'r') as f:
			self.length = len(f['labels'])

	def _get_file(self):
		if self.file == None:
			# Read data from h5py file
			self.h5py_file = h5py.File(self.filename, "r")
			self.firing_times = self.h5py_file["spikes"]["times"]
			self.units_fired = self.h5py_file["spikes"]["units"]
			self.labels = np.array(self.h5py_file["labels"], dtype=int)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		self._get_file()
		times = np.digitize(self.firing_times[index], self.time_bins)
		units = self.units_fired[index]
		length = max(times)

		x_idx = torch.LongTensor(np.array([times, units]))
		x_val = torch.FloatTensor(np.ones(len(times)))
		x_size = torch.Size([length, self.nb_units])

		#x = torch.sparse.FloatTensor(x_idx, x_val, x_size)
		x = torch.sparse_coo_tensor(x_idx, x_val, x_size)
		y = self.labels[index]

		x = x.to_dense()
		T = x.shape[0]
		J = self.nb_units
		Bin = self.num_bins

		# Binning
		with torch.no_grad():
			x = x.contiguous().view(T, J//Bin, Bin).sum(-1)

		# Add zero padding for delayed classification
		if self.delay_steps > 0:
			# Create zero padding: [delay_steps, nb_units_binned]
			zero_padding = torch.zeros(self.delay_steps, x.shape[1], dtype=x.dtype)
			# Concatenate original sequence with zero padding
			x = torch.cat([x, zero_padding], dim=0)

		if self.noise_variance > 0:
			x += torch.poisson(0*x + self.noise_variance,)

		# Create mask: 1 only at the last timestep (where classification should happen)
		# 0 during the original sequence and delay period
		mask = torch.zeros(x.shape[0], dtype=torch.float32)
		mask[-1] = 1.0  # Only the last timestep is valid for classification

		return x, y, mask

	def generateBatch(self, batch):
		xs, ys, masks = zip(*batch)
		xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
		masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
		xlens = torch.tensor([x.shape[0] for x in xs])
		ys = torch.LongTensor(ys)
		return xs, xlens, ys, masks


def load_shd_or_ssc(
	dataset_name,
	data_folder,
	split,
	batch_size,
	nb_steps=100,
	shuffle=True,
	num_workers=0,
	delay_steps=0,
	noise_variance=0,
):
	"""
	This function creates a dataloader for a given split of
	the SHD or SSC datasets with support for delayed classification.

	Arguments
	---------
	dataset_name : str
		Name of the dataset, either shd or ssc.
	data_folder : str
		Path to folder containing the Heidelberg Digits dataset.
	split : str
		Split of dataset, must be either "train" or "test" for SHD.
		For SSC, can be "train", "valid" or "test".
	batch_size : int
		Number of examples in a single generated batch.
	nb_steps : int
		Number of time steps for the generated spike trains.
	shuffle : bool
		Whether to shuffle examples or not.
	num_workers : int
		Number of num_workers.
	delay_steps : int
		Number of zero-padded timesteps to add after the sequence for delayed classification.
		If 0, no padding is added but classification mask still points to the last timestep.
	noise_variance : float
		Variance of additive noise.
		If 0, no noise is added.
	"""
	if dataset_name not in ["shd", "ssc"]:
		raise ValueError(f"Invalid dataset name {dataset_name}")

	if split not in ["train", "valid", "test"]:
		raise ValueError(f"Invalid split name {split}")

	if dataset_name == "shd" and split == "valid":
		logging.info("SHD does not have a validation split. Using test split.")
		split = "test"

	dataset = SpikingDataset(dataset_name, data_folder, split, nb_steps, delay_steps=delay_steps, noise_variance=noise_variance)
	logging.info(f"Number of examples in {split} set: {len(dataset)}")
	if delay_steps > 0:
		logging.info(f"Delayed classification with {delay_steps} delay steps")

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		collate_fn=dataset.generateBatch,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
		prefetch_factor=2,
	)

	return loader


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Test load_shd_or_ssc function")
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="shd",
		choices=["shd", "ssc"],
		help="Dataset name: shd or ssc"
	)
	parser.add_argument(
		"--data_folder",
		type=str,
		required=True,
		help="Path to folder containing the dataset"
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		choices=["train", "valid", "test"],
		help="Dataset split: train, valid, or test"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=32,
		help="Batch size"
	)
	parser.add_argument(
		"--nb_steps",
		type=int,
		default=100,
		help="Number of time steps"
	)
	parser.add_argument(
		"--num_samples",
		type=int,
		default=5,
		help="Number of batches to test"
	)
	parser.add_argument(
		"--delay_steps",
		type=int,
		default=0,
		help="Number of delay steps to add after the sequence (for delayed classification)"
	)
	parser.add_argument(
		"--noise_variance",
		type=float,
		default=0,
		help="Variance of noise added to the signal."
	)

	try:
		args = parser.parse_args()
	except SystemExit:
		import warnings
		args = parser.parse_args([])
		warnings.warn('Could not parse arguments, falling back to default values.')


	# Set up logging
	logging.basicConfig(level=logging.INFO)

	# Load the dataset
	loader = load_shd_or_ssc(
		dataset_name=args.dataset_name,
		data_folder=args.data_folder,
		split=args.split,
		batch_size=args.batch_size,
		nb_steps=args.nb_steps,
		shuffle=True,
		num_workers=0,
		delay_steps=args.delay_steps,
		noise_variance=args.noise_variance,
	)

	# Plot one sample
	import matplotlib.pyplot as plt

	# Get first sample from dataset
	sample_x, sample_y, sample_mask = loader.dataset[0]
	sample_x = sample_x.numpy()
	sample_mask = sample_mask.numpy()

	# Create plot with subplots (mask subplot is much smaller)
	fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
							 gridspec_kw={'height_ratios': [10, 1]})

	# Plot spike raster
	axes[0].imshow(sample_x.T, aspect='auto', interpolation='nearest')
	axes[0].set_ylabel('Neuron unit')
	axes[0].set_title(f'Sample from {args.dataset_name} {args.split} dataset (Label: {sample_y}, Delay: {args.delay_steps} steps)')
	axes[0].axvline(x=len(sample_mask) - args.delay_steps - 1, color='cyan', linestyle='--', linewidth=2, label='End of signal')
	axes[0].legend()

	# Plot mask as imshow (1 row by time steps)
	axes[1].imshow(sample_mask.reshape(1, -1), aspect='auto', vmin=0, vmax=1)
	axes[1].set_ylabel('Classification mask')
	axes[1].set_xlabel('Time step')
	axes[1].set_yticks([])

	plt.tight_layout()
	plt.show()
