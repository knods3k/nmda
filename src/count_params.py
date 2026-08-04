#%%
from utils.diagnosis import count_trainable_parameters, get_n_neurons, get_n_dendrites
from retrieve_model import retrieve_model
from os import listdir

directory = '/Users/cankayser/Downloads/ray_results/'

dir_list = sorted(listdir(directory))
# dir_list.remove(".DS_Store")

for d in dir_list:
	model = retrieve_model(directory + d)
	print(f'{type(model).__name__} {get_n_neurons(model)} {get_n_dendrites(model)} \n {count_trainable_parameters(model)}' )

# %%
