#%%
import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset, DataLoader

from utils.settings import DTYPE, DEVICE_FALLBACK, SEED

CRITERION = torch.nn.MSELoss(reduction='mean')

BATCH_SIZE = 1
N_SPRINGS = 4
T = 200
DELTA_T = 2.5

def off_diagonal(N, k=0):
	I = torch.eye(N, N, device=DEVICE_FALLBACK, dtype=DTYPE)
	I = torch.roll(I, shifts=k, dims=1)
	if k > 0:
		I[:, :k] = 0
	elif k < 0:
		I[:, k:] = 0
	return I

def uniform(low, high, size):
	return (low - high) * torch.rand(size, device=DEVICE_FALLBACK, dtype=DTYPE) + high


class Oscillator():
	def __init__(self, N=N_SPRINGS, delta_t=DELTA_T, T=T):
		self.reset_seed()

		self.N = N
		self.delta_t = delta_t
		self.T = T
		self.t = torch.linspace(0,T,int(T / delta_t) +1, device=DEVICE_FALLBACK, dtype=DTYPE)

		self.zero = torch.zeros((N,N), device=DEVICE_FALLBACK, dtype=DTYPE)
		self.one = torch.eye(N, device=DEVICE_FALLBACK, dtype=DTYPE)

		self.diagonal_idx = torch.where(off_diagonal(N, k=0))
		self.diagonal_idx_up = torch.where(off_diagonal(N, k=1))
		self.diagonal_idx_down = torch.where(off_diagonal(N, k=-1))

		self.s = torch.full((N+1,), torch.nan, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.M = torch.eye(N, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.S = torch.full((self.N, self.N), torch.nan, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.S = self.generate_springs()

		# self.B = np.block([[self.zero, self.one], [-self.S, self.zero]])
		B_top = torch.cat([self.zero, self.one], dim=1)
		B_bottom = torch.cat([-self.S, self.zero], dim=1)
		self.B = torch.cat([B_top, B_bottom], dim=0)

		self.initial_displacement = torch.full((N,), torch.nan, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.initial_velocity = torch.full((N,), torch.nan, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.x0 = torch.full((N*2,), torch.nan, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.x0 = self.generate_initial_condition()

		# self.A = np.block([[self.one, self.zero], [self.zero, self.M]])
		A_top = torch.cat([self.one, self.zero], dim=1)
		A_bottom = torch.cat([self.zero, torch.diag(1/self.M.diagonal())], dim=1)
		self.A_inv = torch.cat([A_top, A_bottom], dim=0)

	@staticmethod
	def reset_seed():
		torch.manual_seed(SEED)

	def generate_springs(self):
		self.s = uniform(0.0005, 0.01, self.N+1)
		self.S = torch.zeros((self.N, self.N), device=DEVICE_FALLBACK, dtype=DTYPE)
		self.S[self.diagonal_idx] = (self.s + torch.roll(self.s, -1))[:-1]
		self.S[self.diagonal_idx_up] = -self.s[:-2]
		self.S[self.diagonal_idx_down] = -self.s[2:]
		return self.S

	def generate_initial_condition(self):
		self.initial_displacement = torch.randn(self.N, device=DEVICE_FALLBACK, dtype=DTYPE)
		self.initial_velocity = 0 * self.initial_displacement
		self.x0[:self.N] = self.initial_displacement
		self.x0[self.N:] = self.initial_velocity
		return self.x0

	def solve(self):
		return torch.matrix_exp((self.A_inv @ self.B) * self.t[:,None,None]) @ self.x0


	def generate_trajectory(self):
		self.x0 = self.generate_initial_condition()
		return self.solve()


class Generator(IterableDataset, Oscillator):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def __iter__(self):
		while True:
			trajectory = self.generate_trajectory()[:,:self.N]
			yield trajectory[:-1], trajectory[1:]


def build_loader(batch_size):
	return DataLoader(Generator(), batch_size=batch_size)

LOADER = build_loader(BATCH_SIZE)

# %%
if __name__ == "__main__":

	oscillator = Oscillator()
	trajectory = oscillator.generate_trajectory()[:,:4]

	def plot_trajectory(trajectory):
		plt.figure(figsize=(8,2))
		for i in range(oscillator.N):
			plt.plot(oscillator.t[:-1], trajectory[:,i], label=rf"$x_{i+1}$")
		plt.legend()
		# plt.savefig('fig')
		plt.show()

	for x, y in LOADER:
		for trajectory in x:
			plot_trajectory(trajectory.numpy())
		break


# %%
