import torch
import torch.nn as nn

def inject_gradient(x, fx, gdx):
	'''
	Replace the gradient of a given function with a surrogate
	(see Sebastian Otte 2024)

	x: Input tensor
	fx: Function evaluated at x
	gdx: Surrogate gradient evaluated at x
	'''

	product = x * gdx.detach()
	return product - product.detach() + fx.detach()

class Surrogate(nn.Module):
	def __init__(self, sigma=1., energy=1., type='gauss', *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.sigma = sigma
		self.energy = energy

		if type == 'gauss':
			self.surrogate = self.gauss

		elif type == 'cauchy':
			self.surrogate = self.cauchy

		elif type == 'laplace':
			self.surrogate = self.laplace

		else:
			raise NotImplementedError('The given surrogate gradient is invalid.')

	def gauss(self, x):
		prefctor = self.energy #/ ((2*torch.pi*(self.sigma**2))**(1/2))
		return prefctor * torch.exp(-(x**2)/ (2*(self.sigma**2)))

	def cauchy(self, x):
		prefactor = self.energy #/ (torch.pi * self.sigma)
		return prefactor / (1 + (x/self.sigma)**2)

	def laplace(self, x):
		prefactor = self.energy #/ (2*self.sigma)
		return prefactor * torch.exp(-(torch.abs(x)) / self.sigma)

	@staticmethod
	def step(x):
		return (x > 0).float()

	def forward(self, x, **kwargs):
		return inject_gradient(x, self.step(x), self.surrogate(x))


torch.serialization.add_safe_globals([Surrogate])
