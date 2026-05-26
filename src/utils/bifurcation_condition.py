#%%
from scipy.special import lambertw as W
import numpy as np

def condition(gam0):
	a = gam0 - .5
	if a == 0:
		return 8
	elif a > 0:
		return 4 + (1/a) * W(4*a*np.exp(-4*a), k=0).real
	elif a < 0:
		return 2/(gam0*(1-gam0))


