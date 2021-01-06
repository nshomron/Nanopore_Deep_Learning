#pytorch
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.optim.optimizer import Optimizer

#other
from random import shuffle as randFunct
from random import randint
import numpy as np
import warnings


def differences_transform(signal):
	return np.diff(signal)

def startMove_transform(signal):
	startPosModification = randint(0, 999)
	return signal[startPosModification: -1000 + startPosModification]

def startMove_transform_test(signal):
	startPosModification = randint(0, 1)
	return signal[startPosModification:-100 + startPosModification]

def cutToWindows_transform(signal, seqLength, stride, winLength):
	splitInput = np.zeros((seqLength, winLength), dtype="int16")
	for i in range(seqLength):
		splitInput[i, :] = signal[(i*stride):(i*stride)+winLength]
	return splitInput

def noise_transform(signal):
	shape = tuple(signal.shape)
	noise = np.random.normal(0,5, size = shape)
	return signal + noise.astype("int16")






# class RangeNormalize(object):
def RangeNormalize(inputs):
	"""
	Given min_val: (R, G, B) and max_val: (R,G,B),
	will normalize each channel of the th.*Tensor to
	the provided min and max values.
	Works by calculating :
		a = (max'-min')/(max-min)
		b = max' - a * max
		new_value = a * value + b
	where min' & max' are given values, 
	and min & max are observed min/max for each channel
	
	Arguments
	---------
	min_range : float or integer
		Min value to which tensors will be normalized
	max_range : float or integer
		Max value to which tensors will be normalized
	fixed_min : float or integer
		Give this value if every sample has the same min (max) and 
		you know for sure what it is. For instance, if you
		have an image then you know the min value will be 0 and the
		max value will be 255. Otherwise, the min/max value will be
		calculated for each individual sample and this will decrease
		speed. Dont use this if each sample has a different min/max.
	fixed_max :float or integer
		See above
	Example:
		>>> x = th.rand(3,5,5)
		>>> rn = RangeNormalize((0,0,10),(1,1,11))
		>>> x_norm = rn(x)
	Also works with just one value for min/max:
		>>> x = th.rand(3,5,5)
		>>> rn = RangeNormalize(0,1)
		>>> x_norm = rn(x)
	"""
	min_val = -1
	max_val = 1
	outputs = []
	_input = inputs
	_min_val = _input.min()
	_max_val = _input.max().data()
	a = (max_val - min_val) / (_max_val - _min_val)
	b = max_val- a * _max_val
	_input = _input.mul(a).add(b)
	outputs.append(_input)
	return _input



## custom data loader -----------------------------------
class NanoporeDataset(Dataset):

	def __init__(self, data, labels, shuffle=False, transform=None):

		self.load_order = list(range(len(labels)))

		if shuffle == True:
			randFunct(self.load_order)
		self.load_order = self.load_order
		self.transform=transform
		self.signal_data=data
		self.labels=labels
		self.numOfReadsLeft = len(labels)


	def __getitem__(self, index):

		itemToGet_index = self.load_order.pop(0)
		itemToGet = self.signal_data[itemToGet_index]
		itemToGet_label = self.labels[itemToGet_index]

		if self.transform is not None:
			itemToGet = self.transform(itemToGet)
		self.numOfReadsLeft = self.numOfReadsLeft -1

		return itemToGet, itemToGet_label


	def __len__(self):
		return len(self.load_order)




string_classes = (str, bytes)


def get_tensor(batch, pin, half=False):
	if isinstance(batch, (np.ndarray, np.generic)):
		batch = T(batch, half=half, cuda=False).contiguous()
		if pin: batch = batch.pin_memory()
		return to_gpu(batch)
	elif isinstance(batch, string_classes):
		return batch
	elif isinstance(batch, collections.Mapping):
		return {k: get_tensor(sample, pin, half) for k, sample in batch.items()}
	elif isinstance(batch, collections.Sequence):
		return [get_tensor(sample, pin, half) for sample in batch]
	raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")



class ReduceLROnPlateau(object):
	"""Reduce learning rate when a metric has stopped improving.
	Models often benefit from reducing the learning rate by a factor
	of 2-10 once learning stagnates. This scheduler reads a metrics
	quantity and if no improvement is seen for a 'patience' number
	of epochs, the learning rate is reduced.
	
	Args:
		factor: factor by which the learning rate will
			be reduced. new_lr = lr * factor
		patience: number of epochs with no improvement
			after which learning rate will be reduced.
		verbose: int. 0: quiet, 1: update messages.
		mode: one of {min, max}. In `min` mode,
			lr will be reduced when the quantity
			monitored has stopped decreasing; in `max`
			mode it will be reduced when the quantity
			monitored has stopped increasing.
		epsilon: threshold for measuring the new optimum,
			to only focus on significant changes.
		cooldown: number of epochs to wait before resuming
			normal operation after lr has been reduced.
		min_lr: lower bound on the learning rate.
		
		
	Example:
		>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
		>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
		>>> for epoch in range(10):
		>>>     train(...)
		>>>     val_acc, val_loss = validate(...)
		>>>     scheduler.step(val_loss, epoch)
	"""

	def __init__(self, optimizer, mode='min', factor=0.5, patience=5000,
				 verbose=1, epsilon=1e-3, cooldown=15000, min_lr=0.00000001):
		super(ReduceLROnPlateau, self).__init__()

		if factor >= 1.0:
			raise ValueError('ReduceLROnPlateau '
							 'does not support a factor >= 1.0.')
		self.factor = factor
		self.min_lr = min_lr
		self.epsilon = epsilon
		self.patience = patience
		self.verbose = verbose
		self.cooldown = cooldown
		self.cooldown_counter = 0  # Cooldown counter.
		self.monitor_op = None
		self.wait = 0
		self.best = 0
		self.mode = mode
		assert isinstance(optimizer, Optimizer)
		self.optimizer = optimizer
		self._reset()

	def _reset(self):
		"""Resets wait counter and cooldown counter.
		"""
		if self.mode not in ['min', 'max']:
			raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
		if self.mode == 'min' :
			self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
			self.best = np.Inf
		else:
			self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
			self.best = -np.Inf
		self.cooldown_counter = 0
		self.wait = 0
		self.lr_epsilon = self.min_lr * 1e-4

	def reset(self):
		self._reset()

	def step(self, metrics, epoch):
		current = metrics
		if current is None:
			warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
		else:
			if self.in_cooldown():
				self.cooldown_counter -= 1
				self.wait = 0

			if self.monitor_op(current, self.best):
				self.best = current
				self.wait = 0
			elif not self.in_cooldown():
				if self.wait >= self.patience:
					for param_group in self.optimizer.param_groups:
						old_lr = float(param_group['lr'])
						if old_lr > self.min_lr + self.lr_epsilon:
							new_lr = old_lr * self.factor
							new_lr = max(new_lr, self.min_lr)
							param_group['lr'] = new_lr
							if self.verbose > 0:
								print('\n\n\n\n\nEpoch %05d: reducing learning rate to %s.\n\n\n\n\n' % (epoch, new_lr))
								# asda
							self.cooldown_counter = self.cooldown
							self.wait = 0
				self.wait += 1

	def in_cooldown(self):
		return self.cooldown_counter > 0