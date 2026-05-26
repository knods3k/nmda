#%%
import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset, DataLoader

from utils.settings import DTYPE, DEVICE, SEED

CE = torch.nn.CrossEntropyLoss()
def sequence_cross_entropy(logits, targets):
    B, T, N_CLASSES = logits.shape
    logits_flat = logits.reshape(-1, N_CLASSES)
    targets_flat = targets.flatten()
    return CE(logits_flat, targets_flat)

CRITERION = sequence_cross_entropy

BATCH_SIZE = 1
N_STATES = 9
T = 99

ENTROPY = 3
CORRELATION_PARAMETER = 9.


class MarkovProcess():
	def __init__(
			self,
			N=N_STATES,
			T=T,
			entropy=ENTROPY,
			correlation_parameter=CORRELATION_PARAMETER):
		self.reset_seed()

		self.N = N
		self.T = T
		self.t = torch.arange(0,T, device=DEVICE)

		self.p = (torch.linspace(1, 0, entropy, device=DEVICE, dtype=DTYPE))**correlation_parameter
		self.P = self.create_transition_matrix(self.N, self.p)

		self.x0 = torch.full((N,), 0., device=DEVICE, dtype=DTYPE)

		self.trajectory = torch.full((T,N), 0., device=DEVICE, dtype=DTYPE)

	@staticmethod
	def reset_seed():
		torch.manual_seed(SEED)

	@staticmethod
	def sample(x):
		idx = torch.multinomial(x, 1)
		x *= 0.
		x[idx] = 1.
		return x

	@staticmethod
	def create_transition_matrix(N, probs):
		assert len(probs) <= N
		P = torch.zeros((N,N), device=DEVICE, dtype=DTYPE)
		probs = probs / probs.sum()

		p = torch.zeros(N, device=DEVICE, dtype=DTYPE)
		p[:len(probs)] = probs
		for i in range(N):
			P[i,:] = torch.roll(p, i)
		return P

	def generate_initial_condition(self):
		self.x0 *= 0
		self.x0 = torch.rand(self.N, device=DEVICE, dtype=DTYPE)
		return self.sample(self.x0)

	def solve(self):
		self.trajectory *= 0
		self.trajectory[0] = self.generate_initial_condition()
		for i in self.t[:-1]:
			self.trajectory[i+1] = self.sample(self.P @ self.trajectory[i])
		return self.trajectory

	def generate_trajectory(self):
		self.x0 = self.generate_initial_condition()
		return self.solve()


class Generator(IterableDataset, MarkovProcess):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def __iter__(self):
		while True:
			trajectory = self.generate_trajectory()[:,:self.N]
			yield trajectory, trajectory.argmax(dim=-1)


def build_loader(batch_size):
	return DataLoader(Generator(), batch_size=batch_size)


LOADER = build_loader(BATCH_SIZE)

# %%
if __name__ == "__main__":

	markov = MarkovProcess()
	trajectory = markov.generate_trajectory()

	def plot_trajectory(trajectory):
		plt.imshow(trajectory.T)
		plt.show()


	for x, y in LOADER:
		for trajectory in x:
			plot_trajectory(trajectory.cpu().numpy())
		break


# %%
