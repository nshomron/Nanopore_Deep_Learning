import torch
import math
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
from torch.nn.init import kaiming_normal, kaiming_uniform, constant





class SeparatedBatchNorm1d(nn.Module):

	"""
	A batch normalization module which keeps its running mean
	and variance separately per timestep.
	"""

	def __init__(self, num_features, max_length, eps=1e-3, momentum=0.1,
				 affine=True):
		"""
		Most parts are copied from
		torch.nn.modules.batchnorm._BatchNorm.
		"""

		super(SeparatedBatchNorm1d, self).__init__()
		self.num_features = num_features
		self.max_length = max_length
		self.affine = affine
		self.eps = eps
		self.momentum = momentum
		if self.affine:
			self.weight = nn.Parameter(torch.FloatTensor(num_features))
			self.bias = nn.Parameter(torch.FloatTensor(num_features))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		for i in range(max_length):
			self.register_buffer(
				'running_mean_{}'.format(i), torch.zeros(num_features))
			self.register_buffer(
				'running_var_{}'.format(i), torch.ones(num_features))
		self.reset_parameters()

	def reset_parameters(self):
		for i in range(self.max_length):
			running_mean_i = getattr(self, 'running_mean_{}'.format(i))
			running_var_i = getattr(self, 'running_var_{}'.format(i))
			running_mean_i.zero_()
			running_var_i.fill_(1)
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def _check_input_dim(self, input_):
		if input_.size(1) != self.running_mean_0.nelement():
			raise ValueError('got {}-feature tensor, expected {}'
							 .format(input_.size(1), self.num_features))

	def forward(self, input_, time):
		self._check_input_dim(input_)
		if time >= self.max_length:
			time = self.max_length - 1
		running_mean = getattr(self, 'running_mean_{}'.format(time))
		running_var = getattr(self, 'running_var_{}'.format(time))
		return functional.batch_norm(
			input=input_, running_mean=running_mean, running_var=running_var,
			weight=self.weight, bias=self.bias, training=self.training,
			momentum=self.momentum, eps=self.eps)

	def __repr__(self):
		return ('{name}({num_features}, eps={eps}, momentum={momentum},'
				' max_length={max_length}, affine={affine})'
				.format(name=self.__class__.__name__, **self.__dict__))


class BNLSTMCell(nn.Module):

	"""A BN-LSTM cell."""

	def __init__(self, input_size, hidden_size, max_length, use_bias=True):

		super(BNLSTMCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.max_length = max_length
		self.use_bias = use_bias
		self.weight_ih = nn.Parameter(
			torch.FloatTensor(input_size, 4 * hidden_size))
		self.weight_hh = nn.Parameter(
			torch.FloatTensor(hidden_size, 4 * hidden_size))
		if use_bias:
			self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
		else:
			self.register_parameter('bias', None)
		# BN parameters
		self.bn_ih = SeparatedBatchNorm1d(
			num_features=4 * hidden_size, max_length=max_length)
		self.bn_hh = SeparatedBatchNorm1d(
			num_features=4 * hidden_size, max_length=max_length)
		self.bn_c = SeparatedBatchNorm1d(
			num_features=hidden_size, max_length=max_length)
		self.reset_parameters()

	def reset_parameters(self):
		"""
		Initialize parameters following the way proposed in the paper.
		"""

		init.orthogonal_(self.weight_ih.data)
		# The hidden-to-hidden weight matrix is initialized as an identity
		# matrix.
		weight_hh_data = torch.eye(self.hidden_size)
		weight_hh_data = weight_hh_data.repeat(1, 4)
		with torch.no_grad():
			self.weight_hh.set_(weight_hh_data)
		# The bias is just set to zero vectors.
		init.constant_(self.bias.data, val=0)
		# Initialization of BN parameters.
		self.bn_ih.reset_parameters()
		self.bn_hh.reset_parameters()
		self.bn_c.reset_parameters()
		self.bn_ih.bias.data.fill_(0)
		self.bn_hh.bias.data.fill_(0)
		self.bn_ih.weight.data.fill_(0.1)
		self.bn_hh.weight.data.fill_(0.1)
		self.bn_c.weight.data.fill_(0.1)

	def forward(self, input_, hx, time):
		"""
		Args:
			input_: A (batch, input_size) tensor containing input
				features.
			hx: A tuple (h_0, c_0), which contains the initial hidden
				and cell state, where the size of both states is
				(batch, hidden_size).
			time: The current timestep value, which is used to
				get appropriate running statistics.
		Returns:
			h_1, c_1: Tensors containing the next hidden and cell state.
		"""

		# print(input_)
		h_0, c_0 = hx
		batch_size = h_0.size(0)
		bias_batch = (self.bias.unsqueeze(0)
					  .expand(batch_size, *self.bias.size()))
		wh = torch.mm(h_0, self.weight_hh)
		# print(input_, self.weight_ih, time)
		wi = torch.mm(input_, self.weight_ih)
		bn_wh = self.bn_hh(wh, time=time)
		bn_wi = self.bn_ih(wi, time=time)
		f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
								 split_size_or_sections=self.hidden_size, dim=1)
		c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
		h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
		# print(h_1)
		return h_1, c_1


class bnLSTM(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	# def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(bnLSTM, self).__init__()
		# self.cell_class = cell_class
		self.real_input_size = input_size
		self.input_size = input_size*8

		self.max_length = int(max_length/8)
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		# self.seqLength = seqLength
		self.conv1d = nn.Conv1d(self.real_input_size , self.input_size, 33, stride=2, padding=16, dilation=1, groups=1, bias=True)
		self.maxpool = nn.MaxPool1d(kernel_size=4, stride=0, padding=0)
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		for layer in range(num_layers):
			layer_input_size = self.input_size  if layer == 0 else hidden_size
			cell = BNLSTMCell(input_size=layer_input_size,
							  hidden_size=hidden_size, max_length = max_length, use_bias=True)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# if isinstance(cell, BNLSTMCell):
			# 	h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# else:
			# 	h_next, c_next = cell(input_=input_[time], hx=hx)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)

		input_ = input_.transpose(0, 1).transpose(1,2)
		input_ = self.conv1d(input_)
		input_ = self.maxpool(input_)
		input_ = functional.relu(input_)
		input_ = input_.transpose(1,2).transpose(0, 1)


		h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		hx = (h0, c0)
		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				device = input_.get_device()
				length = length.cuda(device)
		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
			hx = (hx, hx)
		h_n = []
		c_n = []
		layer_output = None
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			if layer == 0:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM._forward_rnn(
					cell=cell, input_=input_, length=length, hx=hx)
			else:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM._forward_rnn(
					cell=cell, input_=layer_output, length=length, hx=hx)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
			c_n.append(layer_c_n)


		output = layer_output
		output = output[-1]
		output = functional.softmax(self.fc(output), dim=1)
		return output

class bnLSTM_32window(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	# def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(bnLSTM_32window, self).__init__()
		# self.cell_class = cell_class
		self.real_input_size = input_size
		self.input_size = input_size

		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		for layer in range(num_layers):
			layer_input_size = self.input_size  if layer == 0 else hidden_size
			cell = BNLSTMCell(input_size=layer_input_size,
							  hidden_size=hidden_size, max_length = max_length, use_bias=True)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# if isinstance(cell, BNLSTMCell):
			# 	h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# else:
			# 	h_next, c_next = cell(input_=input_[time], hx=hx)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		# print( output, hx)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)


		h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		hx = (h0, c0)
		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				device = input_.get_device()
				length = length.cuda(device)
		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
			hx = (hx, hx)
		h_n = []
		c_n = []
		layer_output = None
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			if layer == 0:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
					cell=cell, input_=input_, length=length, hx=hx)
			else:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
					cell=cell, input_=layer_output, length=length, hx=hx)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
			c_n.append(layer_c_n)


		output = layer_output
		output = output[-1]
		output = functional.softmax(self.fc(output), dim=1)
		return output

class VDCNN_bnLSTM_1window(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	# def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(VDCNN_bnLSTM_1window, self).__init__()
		# self.cell_class = cell_class
		self.VDCNN = VDCNN_withDropout_normalMaxPool_ForRNN(input_size=input_size, hidden_size=hidden_size,\
				 max_length=max_length, n_classes=2, depth=9,\
				 n_fc_neurons=1024, shortcut=False,\
				 dropout=0)
		self.real_input_size = input_size
		input_size = 512
		self.input_size = input_size

		max_length = 59*2
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))

		for layer in range(num_layers):
			layer_input_size = self.input_size  if layer == 0 else hidden_size
			cell = BNLSTMCell(input_size=layer_input_size,
							  hidden_size=hidden_size, max_length = max_length, use_bias=True)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# if isinstance(cell, BNLSTMCell):
			# 	h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# else:
			# 	h_next, c_next = cell(input_=input_[time], hx=hx)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		# print( output, hx)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			test_input = self.VDCNN(input_)
			input_ = test_input.transpose(2,1).transpose(0,1).contiguous()

		h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		hx = (h0, c0)
		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				device = input_.get_device()
				length = length.cuda(device)
		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
			hx = (hx, hx)
		h_n = []
		c_n = []
		layer_output = None
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			if layer == 0:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
					cell=cell, input_=input_, length=length, hx=hx)
			else:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
					cell=cell, input_=layer_output, length=length, hx=hx)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
			c_n.append(layer_c_n)



		h_n = torch.stack(h_n, 0)
		
		output = h_n[0]
		output = functional.softmax(self.fc(output), dim=1)
		return output


class VDCNN_bnLSTM_1window_lastOut(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	# def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(VDCNN_bnLSTM_1window_lastOut, self).__init__()
		# self.cell_class = cell_class
		self.VDCNN = VDCNN_withDropout_normalMaxPool_ForRNN(input_size=input_size, hidden_size=hidden_size,\
				 max_length=max_length, n_classes=2, depth=9,\
				 n_fc_neurons=1024, shortcut=False,\
				 dropout=0)
		self.real_input_size = input_size
		input_size = 512
		self.input_size = input_size

		max_length = 59*2
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))

		for layer in range(num_layers):
			layer_input_size = self.input_size  if layer == 0 else hidden_size
			cell = BNLSTMCell(input_size=layer_input_size,
							  hidden_size=hidden_size, max_length = max_length, use_bias=True)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# if isinstance(cell, BNLSTMCell):
			# 	h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# else:
			# 	h_next, c_next = cell(input_=input_[time], hx=hx)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		# print( output, hx)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			test_input = self.VDCNN(input_)
			input_ = test_input.transpose(2,1).transpose(0,1).contiguous()

		h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		hx = (h0, c0)
		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				device = input_.get_device()
				length = length.cuda(device)
		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
			hx = (hx, hx)
		h_n = []
		c_n = []
		layer_output = None
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			if layer == 0:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
					cell=cell, input_=input_, length=length, hx=hx)
			else:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window._forward_rnn(
					cell=cell, input_=layer_output, length=length, hx=hx)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
			c_n.append(layer_c_n)


		output = layer_output


		output = output[-1]

		output = functional.softmax(self.fc(output), dim=1)
		return output

class VDCNN_gru_1window_hidden(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(VDCNN_gru_1window_hidden, self).__init__()
		self.VDCNN = VDCNN_withDropout_normalMaxPool_ForRNN(input_size=input_size, hidden_size=hidden_size,\
				 max_length=max_length, n_classes=2, depth=9,\
				 n_fc_neurons=1024, shortcut=False,\
				 dropout=0)
		self.real_input_size = input_size
		input_size = 512
		self.input_size = input_size

		max_length = 59*2
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))


		self.batchnorm0 = nn.BatchNorm1d(input_size, momentum  = 0.5)
		self.gru = nn.GRU(input_size=self.input_size,
							  hidden_size=hidden_size, bias =True, bidirectional = True)



	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# if isinstance(cell, BNLSTMCell):
			# 	h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# else:
			# 	h_next, c_next = cell(input_=input_[time], hx=hx)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			test_input = self.VDCNN(input_)
			input_ = test_input.transpose(2,1).transpose(0,1).contiguous()
	


		input_ = input_.transpose(0, 1).transpose(1,2).contiguous()
		input_= self.batchnorm0(input_)
		input_ = input_.transpose(1,2).transpose(0, 1).contiguous()
		_ , output2= self.gru(input_)




		output = output2[0]

		output = functional.softmax(self.fc(output), dim=1)
		return output


class VDCNN_gru_1window_lastStep(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	# def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(VDCNN_gru_1window_lastStep, self).__init__()
		self.VDCNN = VDCNN_withDropout_normalMaxPool_ForRNN(input_size=input_size, hidden_size=hidden_size,\
				 max_length=max_length, n_classes=2, depth=9,\
				 n_fc_neurons=1024, shortcut=False,\
				 dropout=0)
		self.real_input_size = input_size
		input_size = 512
		self.input_size = input_size

		max_length = 59*2
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size*2, out_features=int(num_classes))


		self.batchnorm0 = nn.BatchNorm1d(input_size, momentum  = 0.5)
		self.gru = nn.GRU(input_size=self.input_size,
							  hidden_size=hidden_size, bias =True, bidirectional = True)



	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			test_input = self.VDCNN(input_)
			input_ = test_input.transpose(2,1).transpose(0,1).contiguous()
	


		input_ = input_.transpose(0, 1).transpose(1,2).contiguous()
		input_= self.batchnorm0(input_)
		input_ = input_.transpose(1,2).transpose(0, 1).contiguous()
		output2 , _= self.gru(input_)

		output = output2[-1]

		output = functional.softmax(self.fc(output), dim=1)
		return output


class regLSTM_32window(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(regLSTM_32window, self).__init__()
		self.real_input_size = input_size
		self.input_size = input_size

		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		self.lstm = nn.LSTM(input_size=self.input_size,
							  hidden_size=hidden_size, bias =True, bidirectional = False)

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)

		output2 , _= self.lstm(input_)

		output = output2[-1]
		output = functional.softmax(self.fc(output), dim=1)
		return output




class regLSTM_32window_hidden(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2, bidirectional = False):
		super(regLSTM_32window_hidden, self).__init__()
		self.real_input_size = input_size
		self.input_size = input_size

		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		self.lstm = nn.LSTM(input_size=self.input_size,
							  hidden_size=hidden_size, bias =True, bidirectional = bidirectional)

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)
		_ , output2= self.lstm(input_)

		output = output2[0]
		output = output[0]
		output = functional.softmax(self.fc(output), dim=1)
		return output

class regLSTM_32window_hidden_BN(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2, bidirectional = False):
		super(regLSTM_32window_hidden_BN, self).__init__()
		self.real_input_size = input_size
		self.input_size = input_size

		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		self.batchnorm0 = nn.BatchNorm1d(input_size, momentum  = 0.5)
		self.lstm = nn.LSTM(input_size=self.input_size,
							  hidden_size=hidden_size, bias =True, bidirectional = bidirectional)

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)
		input_ = input_.transpose(0, 1).transpose(1,2).contiguous()
		input_= self.batchnorm0(input_)
		input_ = input_.transpose(1,2).transpose(0, 1).contiguous()
		_ , output2= self.lstm(input_)
		output = output2[0]
		output = output[0]
		output = functional.softmax(self.fc(output), dim=1)
		return output


class regGru_32window_hidden_BN(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2, bidirectional = False):
		super(regGru_32window_hidden_BN, self).__init__()
		self.real_input_size = input_size
		self.input_size = input_size

		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))

		self.batchnorm0 = nn.BatchNorm1d(input_size, momentum  = 0.5)
		self.gru = nn.GRU(input_size=self.input_size,
							  hidden_size=hidden_size, bias =True, bidirectional = bidirectional)


	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()
	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)
		input_ = input_.transpose(0, 1).transpose(1,2).contiguous()
		input_= self.batchnorm0(input_)
		input_ = input_.transpose(1,2).transpose(0, 1).contiguous()
		_ , output2= self.gru(input_)
		output = output2
		output = output[0]
		output = functional.softmax(self.fc(output), dim=1)
		return output



class bnLSTM_32window_h_n(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, input_size, hidden_size,max_length,  num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 2):
		super(bnLSTM_32window_h_n, self).__init__()
		self.real_input_size = input_size
		self.input_size = input_size

		self.max_length = max_length
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		for layer in range(num_layers):
			layer_input_size = self.input_size  if layer == 0 else hidden_size
			cell = BNLSTMCell(input_size=layer_input_size,
							  hidden_size=hidden_size, max_length = max_length, use_bias=True)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			c_next = c_next*mask + hx[1]*(1 - mask)
			hx_next = (h_next, c_next)
			output.append(h_next)
			hx = hx_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)
		h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
		hx = (h0, c0)
		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				device = input_.get_device()
				length = length.cuda(device)
		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
			hx = (hx, hx)
		h_n = []
		c_n = []
		layer_output = None
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			if layer == 0:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window_h_n._forward_rnn(
					cell=cell, input_=input_, length=length, hx=hx)
			else:
				layer_output, (layer_h_n, layer_c_n) = bnLSTM_32window_h_n._forward_rnn(
					cell=cell, input_=layer_output, length=length, hx=hx)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
			c_n.append(layer_c_n)


		output = layer_output
		h_n = torch.stack(h_n, 0)
		output = h_n[0]
		output = functional.softmax(self.fc(output), dim=1)
		return output




class RecurrentLSTM(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(RecurrentLSTM, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.model_vdcnn = VDCNN(input_size=winLength,
			 hidden_size=input_size, max_length = seqLength, num_layers=1,\
			 dropout=0, num_classes = 2 ,batch_first = True, \
			 n_classes=2, num_embedding=int(seqLength), embedding_dim=16, depth=9,
			 n_fc_neurons=128, shortcut=True, lengthOfFlagLenMore = seqLength)

		self.model_gru = torch.nn.GRU(input_size = input_size, \
			hidden_size = hidden_size,\
			 num_layers = 2, bias = True, batch_first = False,\
			 dropout = dropout, bidirectional = False) 
		self.model = LSTM(cell_class=cell_class, input_size=input_size,
					 hidden_size=hidden_size, batch_first=batch_first,
					 num_layers=num_layers,  dropout=dropout, num_classes = num_classes, seqLength = 64, winLength = winLength, outChannele = outChannele, **kwargs)
		self.model2 = LSTM(cell_class=cell_class, input_size=hidden_size*2,
					 hidden_size=hidden_size, batch_first=batch_first,
					 num_layers=num_layers,  dropout=0, num_classes = num_classes, seqLength = int(((seqLength-64)-64)/4), winLength = winLength, outChannele = outChannele, **kwargs)
		self.model3 = LSTM(cell_class=cell_class, input_size=hidden_size*4,
					 hidden_size=num_classes, batch_first=batch_first,
					 num_layers=num_layers,  dropout=0, num_classes = num_classes, seqLength = int(((((seqLength-64)-64)/4)-64)/4), winLength = winLength, outChannele = outChannele, **kwargs)

		self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
		self.adaptmaxpool = nn.AdaptiveMaxPool1d(64)
		self.batchnorm = nn.BatchNorm1d(winLength, eps=1e-05, momentum=0.1, affine=True)
		self.conv1d1 = nn.Conv1d(winLength, input_size, 65, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 65, stride=4, padding=0, dilation=1, groups=1, bias=True)

		self.conv1d3 = nn.Conv1d(hidden_size, hidden_size*4, 65, stride=4, padding=0, dilation=1, groups=1, bias=True)

	def forward(self, input_, length=None, hx=None):
		h0 = Variable(input_.data.new(input_.size(0), self.hidden_size)
					  .normal_(0, 0.1))
		c0 = Variable(input_.data.new(input_.size(0), self.hidden_size)
					  .normal_(0, 0.1))
		hx = (h0, c0)
		output = input_.transpose(1,2).contiguous()
		output = self.conv1d1(output)
		output = self.adaptmaxpool(output)
		output = output.transpose(2,1).transpose(1,0)
		output, _ = self.model(input_=output, hx=hx)
		output = output.transpose(0,1)[:,-1,:]
		return functional.softmax(self.fc(output),dim=1)


class BasicConvResBlock(nn.Module):

	def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
		super(BasicConvResBlock, self).__init__()

		self.downsample = downsample
		self.shortcut = shortcut

		self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
		self.bn1 = nn.BatchNorm1d(n_filters)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
		self.bn2 = nn.BatchNorm1d(n_filters)

	def forward(self, x):

		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.shortcut:
			if self.downsample is not None:
				residual = self.downsample(x)
			out += residual

		out = self.relu(out)

		return out


class VDCNN_noDropout(nn.Module):

	def __init__(self, input_size, hidden_size,\
	 max_length, n_classes=2, depth=9,\
	 n_fc_neurons=2048, shortcut=False,   \
	 dropout=0.5\
	 ):
		super(VDCNN_noDropout, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.max_length = max_length
		self.num_directions = 1

		layers = []
		fc_layers = []

		layers.append(nn.Conv1d(input_size, 64, kernel_size=3, padding=1))

		if depth == 9:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
		elif depth == 17:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
		elif depth == 29:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
		elif depth == 49:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

		layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		for _ in range(n_conv_block_64-1):
			layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 2

		ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
		layers.append(BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_128-1):
			layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 4

		ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
		layers.append(BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_256 - 1):
			layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

		ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
		layers.append(BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_512 - 1):
			layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))

		last_pooling_layer = 'k-max-pooling'
		if last_pooling_layer == 'k-max-pooling':
			layers.append(nn.AdaptiveMaxPool1d(32))
			fc_layers.extend([nn.Linear(32*512 + 0, n_fc_neurons), nn.ReLU()])
		elif last_pooling_layer == 'max-pooling':
			layers.append(nn.MaxPool1d(kernel_size=8, stride=2, padding=0))
			fc_layers.extend([nn.Linear(61*512, n_fc_neurons), nn.ReLU()])
		else:
			raise

		fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
		fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)
		self.flagDropout = nn.Dropout(p = dropout, inplace = False)

		self.__init_weights()

	def __init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				kaiming_normal(m.weight, mode='fan_in')
				if m.bias is not None:
					constant(m.bias, 0)

	def forward(self, input_):

		batchSize = input_.size(0)
		out = input_.transpose(1,2).contiguous()
		out = self.layers(out)
		out = out.view(out.size(0), -1)
		out = self.fc_layers(out)
		return out

class VDCNN_noDropout_normalMaxPool(nn.Module):

	def __init__(self, input_size, hidden_size,\
	 max_length, n_classes=2, depth=9,\
	 n_fc_neurons=2048, shortcut=False,   \
	 dropout=0.5\
	 ):
		super(VDCNN_noDropout_normalMaxPool, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.max_length = max_length
		self.num_directions = 1

		layers = []
		fc_layers = []
		layers.append(nn.Conv1d(input_size, 64, kernel_size=3, padding=1))

		if depth == 9:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
		elif depth == 17:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
		elif depth == 29:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
		elif depth == 49:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

		layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		for _ in range(n_conv_block_64-1):
			layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 2

		ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
		layers.append(BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_128-1):
			layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 4

		ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
		layers.append(BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_256 - 1):
			layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

		ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
		layers.append(BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_512 - 1):
			layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))

		last_pooling_layer = 'max-pooling'
		if last_pooling_layer == 'k-max-pooling':
			layers.append(nn.AdaptiveMaxPool1d(32))
			fc_layers.extend([nn.Linear(32*512 + 0, n_fc_neurons), nn.ReLU()])
		elif last_pooling_layer == 'max-pooling':
			layers.append(nn.MaxPool1d(kernel_size=16, stride=16, padding=0))
			fc_layers.extend([nn.Linear(15*512, n_fc_neurons), nn.ReLU()])
		else:
			raise

		fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
		fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)

		self.flagDropout = nn.Dropout(p = dropout, inplace = False)

		self.__init_weights()

	def __init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				kaiming_normal(m.weight, mode='fan_in')
				if m.bias is not None:
					constant(m.bias, 0)

	def forward(self, input_):

		batchSize = input_.size(0)

		out = input_.transpose(1,2).contiguous()
		out = self.layers(out)
		out = out.view(out.size(0), -1)
		out = self.fc_layers(out)
		return out


class VDCNN_withDropout_normalMaxPool(nn.Module):

	def __init__(self, input_size, hidden_size,\
	 max_length, n_classes=2, depth=9,\
	 n_fc_neurons=2048, shortcut=False,   \
	 dropout=0.5\
	 ):
		super(VDCNN_withDropout_normalMaxPool, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.max_length = max_length
		self.num_directions = 1

		layers = []
		fc_layers = []


		layers.append(nn.Conv1d(input_size, 64, kernel_size=3, padding=1))

		if depth == 9:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
		elif depth == 17:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
		elif depth == 29:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
		elif depth == 49:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

		layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		for _ in range(n_conv_block_64-1):
			layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 2

		ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
		layers.append(BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_128-1):
			layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) # l = initial length / 4

		ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
		layers.append(BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_256 - 1):
			layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

		ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
		layers.append(BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_512 - 1):
			layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))

		last_pooling_layer = 'max-pooling'
		if last_pooling_layer == 'k-max-pooling':
			layers.append(nn.AdaptiveMaxPool1d(32))
			fc_layers.extend([nn.Linear(32*512 + 0, n_fc_neurons), nn.ReLU()])
		elif last_pooling_layer == 'max-pooling':
			layers.append(nn.MaxPool1d(kernel_size=16, stride=16, padding=0))
			fc_layers.extend([nn.Linear(15*512, n_fc_neurons), nn.ReLU()])
		else:
			raise

		fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
		fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)
		self.flagDropout = nn.Dropout(p = dropout, inplace = False)

		self.__init_weights()

	def __init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				kaiming_normal(m.weight, mode='fan_in')
				# print(m)
				# print(m.bias)
				if m.bias is not None:
					constant(m.bias, 0)

	def forward(self, input_):

		batchSize = input_.size(0)
		out = input_.transpose(1,2).contiguous()
		out = self.layers(out)
		out = out.view(out.size(0), -1)
		out = self.flagDropout(out)
		out = self.fc_layers(out)

		return out


class VDCNN_withDropout_normalMaxPool_ForRNN(nn.Module):

	def __init__(self, input_size, hidden_size,\
	 max_length, n_classes=2, depth=9,\
	 n_fc_neurons=2048, shortcut=False,   \
	 dropout=0.5\
	 ):
		super(VDCNN_withDropout_normalMaxPool_ForRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.max_length = max_length
		self.num_directions = 1

		layers = []
		fc_layers = []


		layers.append(nn.Conv1d(input_size, 64, kernel_size=3, padding=1))

		if depth == 9:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
		elif depth == 17:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
		elif depth == 29:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
		elif depth == 49:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

		layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		for _ in range(n_conv_block_64-1):
			layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1)) # l = initial length / 2

		ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
		layers.append(BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_128-1):
			layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1)) # l = initial length / 4

		ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
		layers.append(BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_256 - 1):
			layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

		ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
		layers.append(BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_512 - 1):
			layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=3, padding=1, shortcut=shortcut))

		last_pooling_layer = 'max-pooling'
		if last_pooling_layer == 'k-max-pooling':
			layers.append(nn.AdaptiveMaxPool1d(32))
			fc_layers.extend([nn.Linear(32*512 + 0, n_fc_neurons), nn.ReLU()])
		elif last_pooling_layer == 'max-pooling':
			layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
			fc_layers.extend([nn.Linear(15*512, n_fc_neurons), nn.ReLU()])
		else:
			raise

		fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])

		self.layers = nn.Sequential(*layers)
		self.flagDropout = nn.Dropout(p = dropout, inplace = False)

		self.__init_weights()

	def __init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				kaiming_normal(m.weight, mode='fan_in')
				if m.bias is not None:
					constant(m.bias, 0)

	def forward(self, input_):

		batchSize = input_.size(0)
		out = input_.transpose(1,2).contiguous()
		out = self.layers(out)
		return out



class VDCNN_noDropout_normalMaxPool_largeKernel(nn.Module):

	def __init__(self, input_size, hidden_size,\
	 max_length, n_classes=2, depth=9,\
	 n_fc_neurons=2048, shortcut=False,   \
	 dropout=0.5\
	 ):
		super(VDCNN_noDropout_normalMaxPool_largeKernel, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.max_length = max_length
		self.num_directions = 1

		layers = []
		fc_layers = []


		layers.append(nn.Conv1d(input_size, 64, kernel_size=3, padding=1))

		if depth == 9:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
		elif depth == 17:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
		elif depth == 29:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
		elif depth == 49:
			n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

		layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=65, padding=32, shortcut=shortcut))
		for _ in range(n_conv_block_64-1):
			layers.append(BasicConvResBlock(input_dim=64, n_filters=64, kernel_size=33, padding=16, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=5, stride=3, padding=2)) # l = initial length / 2

		ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
		layers.append(BasicConvResBlock(input_dim=64, n_filters=128, kernel_size=65, padding=32, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_128-1):
			layers.append(BasicConvResBlock(input_dim=128, n_filters=128, kernel_size=33, padding=16, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=7, stride=4, padding=3)) # l = initial length / 4

		ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
		layers.append(BasicConvResBlock(input_dim=128, n_filters=256, kernel_size=17, padding=8, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_256 - 1):
			layers.append(BasicConvResBlock(input_dim=256, n_filters=256, kernel_size=7, padding=3, shortcut=shortcut))
		layers.append(nn.MaxPool1d(kernel_size=7, stride=4, padding=3))

		ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
		layers.append(BasicConvResBlock(input_dim=256, n_filters=512, kernel_size=7, padding=3, shortcut=shortcut, downsample=ds))
		for _ in range(n_conv_block_512 - 1):
			layers.append(BasicConvResBlock(input_dim=512, n_filters=512, kernel_size=7, padding=3, shortcut=shortcut))

		last_pooling_layer = 'max-pooling'
		if last_pooling_layer == 'k-max-pooling':
			layers.append(nn.AdaptiveMaxPool1d(32))
			fc_layers.extend([nn.Linear(32*512 + 0, n_fc_neurons), nn.ReLU()])
		elif last_pooling_layer == 'max-pooling':
			layers.append(nn.MaxPool1d(kernel_size=16, stride=8, padding=0))
			fc_layers.extend([nn.Linear(4*512, n_fc_neurons), nn.ReLU()])
		else:
			raise

		fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
		fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)

		self.flagDropout = nn.Dropout(p = dropout, inplace = False)

		self.__init_weights()

	def __init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				kaiming_normal(m.weight, mode='fan_in')
				if m.bias is not None:
					constant(m.bias, 0)

	def forward(self, input_):

		batchSize = input_.size(0)

		out = input_.transpose(1,2).contiguous()
		out = self.layers(out)


		out = out.view(out.size(0), -1)

		out = self.fc_layers(out)

		return out



class simpleCNN_3Layers_noDilation_largeKernel_noDropout(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(simpleCNN_3Layers_noDilation_largeKernel_noDropout, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.batchnorm0 = nn.BatchNorm1d(winLength, momentum  = 0.5)
		self.conv1d1 = nn.Conv1d(winLength, hidden_size, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)

		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 32, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d3 = nn.Conv1d(hidden_size*2, hidden_size*4, 64, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.fc = nn.Linear(in_features=int(hidden_size*4*44), out_features=int(num_classes))

	def forward(self, input_, length=None, hx=None):
		output = input_.transpose(1,2).contiguous()
		output = self.batchnorm0(output)
		output = self.conv1d1(output)
		output = self.maxpool1(output)
		output = functional.relu(output)
		output = self.conv1d2(output)
		output = self.maxpool2(output)
		output = functional.relu(output)
		output = self.conv1d3(output)
		output = self.maxpool3(output)
		output = functional.relu(output)
		return functional.softmax(self.fc(output.view(output.size(0), -1)),dim=1)

class simpleCNN_3Layers_noDilation_largeKernel_withDropout(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(simpleCNN_3Layers_noDilation_largeKernel_withDropout, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.batchnorm0 = nn.BatchNorm1d(winLength, momentum  = 0.5)
		self.conv1d1 = nn.Conv1d(winLength, hidden_size, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)

		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 32, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d3 = nn.Conv1d(hidden_size*2, hidden_size*4, 64, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.fc = nn.Linear(in_features=int(hidden_size*4*44), out_features=int(num_classes))
		self.Dropout1 = nn.Dropout(p = dropout, inplace = False)

	def forward(self, input_, length=None, hx=None):
		output = input_.transpose(1,2).contiguous()
		output = self.batchnorm0(output)
		output = self.conv1d1(output)
		output = self.maxpool1(output)
		output = functional.relu(output)
		output = self.conv1d2(output)
		output = self.maxpool2(output)
		output = functional.relu(output)
		output = self.conv1d3(output)
		output = self.maxpool3(output)
		output = functional.relu(output)
		output = self.Dropout1(output)
		return functional.softmax(self.fc(output.view(output.size(0), -1)),dim=1)

class simpleCNN_3Layers_withDilation_largeKernel_withDropout(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(simpleCNN_3Layers_withDilation_largeKernel_withDropout, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.batchnorm0 = nn.BatchNorm1d(winLength, momentum  = 0.5)
		self.conv1d1 = nn.Conv1d(winLength, hidden_size, 16, stride=1, padding=0, dilation=2, groups=1, bias=True)
		
		self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)

		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 32, stride=2, padding=0, dilation=2, groups=1, bias=True)
		self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d3 = nn.Conv1d(hidden_size*2, hidden_size*4, 64, stride=2, padding=0, dilation=2, groups=1, bias=True)
		self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.fc = nn.Linear(in_features=int(hidden_size*4*26), out_features=int(num_classes))
		self.Dropout1 = nn.Dropout(p = dropout, inplace = False)

	def forward(self, input_, length=None, hx=None):
		output = input_.transpose(1,2).contiguous()
		output = self.batchnorm0(output)
		output = self.conv1d1(output)
		output = self.maxpool1(output)
		output = functional.relu(output)
		output = self.conv1d2(output)
		output = self.maxpool2(output)
		output = functional.relu(output)
		output = self.conv1d3(output)
		output = self.maxpool3(output)
		output = functional.relu(output)
		output = self.Dropout1(output)
		return functional.softmax(self.fc(output.view(output.size(0), -1)),dim=1)

class simpleCNN_3Layers_noDilation_smallKernel_withDropout(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(simpleCNN_3Layers_noDilation_smallKernel_withDropout, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.batchnorm0 = nn.BatchNorm1d(winLength, momentum  = 0.5)
		self.conv1d1 = nn.Conv1d(winLength, hidden_size, 7, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)

		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 7, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d3 = nn.Conv1d(hidden_size*2, hidden_size*4, 7, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.fc = nn.Linear(in_features=int(hidden_size*4*60), out_features=int(num_classes))
		self.Dropout1 = nn.Dropout(p = dropout, inplace = False)

	def forward(self, input_, length=None, hx=None):
		output = input_.transpose(1,2).contiguous()
		output = self.batchnorm0(output)
		output = self.conv1d1(output)
		output = self.maxpool1(output)
		output = functional.relu(output)
		output = self.conv1d2(output)
		output = self.maxpool2(output)
		output = functional.relu(output)
		output = self.conv1d3(output)
		output = self.maxpool3(output)
		output = functional.relu(output)
		output = self.Dropout1(output)
		return functional.softmax(self.fc(output.view(output.size(0), -1)),dim=1)

class simpleCNN_6Layers_noDilation_largeKernel_withDropout(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(simpleCNN_6Layers_noDilation_largeKernel_withDropout, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.batchnorm0 = nn.BatchNorm1d(winLength, momentum  = 0.5)
		self.conv1d1 = nn.Conv1d(winLength, hidden_size, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)

		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 32, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d3 = nn.Conv1d(hidden_size*2, hidden_size*4, 64, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d4 = nn.Conv1d(hidden_size*4, hidden_size*8, 32, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d5 = nn.Conv1d(hidden_size*8, hidden_size*8, 32, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d6 = nn.Conv1d(hidden_size*8, hidden_size*8, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.fc = nn.Linear(in_features=int(hidden_size*8*43), out_features=int(num_classes))
		self.Dropout1 = nn.Dropout(p = dropout, inplace = False)

	def forward(self, input_, length=None, hx=None):
		output = input_.transpose(1,2).contiguous()
		output = self.batchnorm0(output)
		output = self.conv1d1(output)
		output = self.maxpool1(output)
		output = functional.relu(output)
		output = self.conv1d2(output)
		output = self.maxpool2(output)
		output = functional.relu(output)
		output = self.conv1d3(output)
		output = self.maxpool3(output)
		output = functional.relu(output)
		output = self.conv1d4(output)
		output = functional.relu(output)
		output = self.conv1d5(output)
		output = functional.relu(output)
		output = self.conv1d6(output)
		output = functional.relu(output)
		output = self.Dropout1(output)
		return functional.softmax(self.fc(output.view(output.size(0), -1)),dim=1)

class simpleCNN_10Layers_noDilation_largeKernel_withDropout(nn.Module):

	"""A module that runs multiple steps of LSTM."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0.5,num_classes = 25,seqLength = 256,winLength = 128,
				 outChannele = 128,  **kwargs):
		super(simpleCNN_10Layers_noDilation_largeKernel_withDropout, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.seqLength = seqLength
		self.num_classes = num_classes
		self.batchnorm0 = nn.BatchNorm1d(winLength, momentum  = 0.5)
		self.conv1d1 = nn.Conv1d(winLength, hidden_size, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)

		self.conv1d2 = nn.Conv1d(hidden_size, hidden_size*2, 32, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d3 = nn.Conv1d(hidden_size*2, hidden_size*4, 64, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=0, padding=0)
		self.conv1d4 = nn.Conv1d(hidden_size*4, hidden_size*8, 32, stride=2, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d5 = nn.Conv1d(hidden_size*8, hidden_size*8, 32, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d6 = nn.Conv1d(hidden_size*8, hidden_size*8, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d7 = nn.Conv1d(hidden_size*8, hidden_size*8, 8, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d8 = nn.Conv1d(hidden_size*8, hidden_size*8, 8, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d9 = nn.Conv1d(hidden_size*8, hidden_size*8, 8, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1d10 = nn.Conv1d(hidden_size*8, hidden_size*8, 8,  stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.fc = nn.Linear(in_features=int(hidden_size*4*43), out_features=int(num_classes))
		self.Dropout1 = nn.Dropout(p = dropout, inplace = False)

	def forward(self, input_, length=None, hx=None):
		output = input_.transpose(1,2).contiguous()
		output = self.batchnorm0(output)
		output = self.conv1d1(output)
		output = self.maxpool1(output)
		output = functional.relu(output)
		output = self.conv1d2(output)
		output = self.maxpool2(output)
		output = functional.relu(output)
		output = self.conv1d3(output)
		output = self.maxpool3(output)
		output = functional.relu(output)
		output = self.conv1d4(output)
		output = functional.relu(output)
		output = self.conv1d5(output)
		output = functional.relu(output)
		output = self.conv1d6(output)
		output = functional.relu(output)
		output = self.Dropout1(output)
		return functional.softmax(self.fc(output.view(output.size(0), -1)),dim=1)
