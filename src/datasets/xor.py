#%%
import torch

from torch.utils.data import DataLoader, IterableDataset, Dataset

from utils.settings import NUM_WORKERS, PREFETCH_FACTOR
from config.ray.pbt import CONFIG

CONFIG['n_inputs'] = 20
CONFIG['n_outputs'] = 2
# CONFIG['evaluation'] = 'accuracy_last'

torch.manual_seed(42)


CE = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
def sequence_cross_entropy(logits, targets):
    B, T, N_CLASSES = logits.shape
    logits_flat = logits.reshape(-1, N_CLASSES)
    targets_flat = targets.unsqueeze(-1).expand(B, T).reshape(-1)
    return CE(logits_flat, targets_flat)

CRITERION = sequence_cross_entropy



TIME_STEPS =200 #delyad timesteps
CHANNEL_RATE = [0.2,0.6] #spiking rates of high or low
NOISE_RATE = 0.01
CHANNEL_SIZE = 20
#lasting time of signal
CODING_TIME = 10
TEST_TIME = 1

perm = torch.randperm(len(CHANNEL_RATE)**2)
index = perm[:len(CHANNEL_RATE)**2//2]
label = torch.zeros(len(CHANNEL_RATE),len(CHANNEL_RATE))
label[1][0] = 1
label[0][1] = 1


class SpikingDataset(Dataset):
	"""

	Arguments
	---------
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
		batch_size,
		time_steps=TIME_STEPS,
		channel_rate=CHANNEL_RATE,
		noise_rate=NOISE_RATE,
		channel_size=CHANNEL_SIZE,
		coding_time=CODING_TIME,
		test_time=TEST_TIME,
	):

		self.time_steps = time_steps
		self.channel_rate = channel_rate
		self.noise_rate = noise_rate
		self.channel_size = channel_size
		self.coding_time = coding_time
		self.test_time = test_time

	def __len__(self):
		return self.time_steps

	def __getitem__(self, index):
		x,y,z = get_batch(
			1,
			time_steps=TIME_STEPS,
			channel_rate=CHANNEL_RATE,
			noise_rate=NOISE_RATE,
			channel_size=CHANNEL_SIZE,
			coding_time=CODING_TIME,
			test_time=TEST_TIME,
			)
		return x[0],y[0],z[0]



def get_batch(batch_size,
			  time_steps=TIME_STEPS,
			  channel_rate=CHANNEL_RATE,
			  noise_rate=NOISE_RATE,
			  channel_size=CHANNEL_SIZE,
			  coding_time=CODING_TIME,
			  test_time=TEST_TIME,
			  ):
    """Generate the delayed spiking xor problem dataset"""

    values = torch.rand(batch_size,time_steps,channel_size,requires_grad=False) <= noise_rate
    targets = torch.zeros(time_steps,batch_size,requires_grad=False).int()
    #generate the first signal
    init_pattern = torch.randint(len(channel_rate),size=(batch_size,))
    prob_matrix = torch.ones(coding_time,channel_size,batch_size)*torch.tensor(channel_rate)[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()

    values[:,:coding_time,:] = values[:,:coding_time,:] | add_patterns
    #generate the position of delayed signal
    position = torch.randint(test_time,size=(batch_size,))
    pattern = torch.randint(len(channel_rate),size=(batch_size,))
    label_t = label[init_pattern,pattern].int()
    prob = torch.tensor(channel_rate)[pattern]
    prob_matrix = torch.ones(coding_time,channel_size,batch_size)*prob
    add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()
    #generate the delayed signal
    for i in range(batch_size):
        values[i,time_steps-(position[i]+1)*coding_time:time_steps-(position[i])*coding_time,:] = values[i,time_steps-(position[i]+1)*coding_time:time_steps-(position[i])*coding_time,:] | add_patterns[i]
        targets[time_steps-(position[i]+1)*coding_time:,i] = label_t[i]


    return values, targets.transpose(0,1).contiguous(),position

class Generator(IterableDataset):
	def __init__(self, dataset: SpikingDataset, shuffle=True, **kwargs):
		self.dataset = dataset
		self.shuffle = shuffle
		self.N_samples = len(self.dataset)

	def generateBatch(self, batch):
		xs, ys, mask = zip(*batch)
		xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
		ys = torch.stack(ys)
		return xs.float(), ys

	def __iter__(self):
		if self.shuffle:
			while True:
				idx = torch.randint(0, self.N_samples, ()).item()
				yield self.dataset[idx]
		else:
			while True:
				for idx in range(self.N_samples):
					yield self.dataset[idx]

# %%
def build(batch_size, split='train', shuffle=True, **kwargs):
    base = SpikingDataset(batch_size)
    iterable = Generator(base)

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
