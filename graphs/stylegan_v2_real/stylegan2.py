from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import random
import math
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch import nn, optim

from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torchvision
from .networks import Generator, Discriminator

class StyleGAN():

	def __init__(self, lr):
		size = 256
		code_size = 512
		n_mlp = 8
		channel_multiplier = 2
		self.netG = Generator(size, code_size, n_mlp, channel_multiplier=channel_multiplier).cuda()
		self.netD = Discriminator(size,  channel_multiplier=channel_multiplier).cuda()
		self.g_running = Generator(size, code_size, n_mlp, channel_multiplier=channel_multiplier).cuda()
		self.g_running.train(False)

		self.learningRate = lr
		self.optimizerD = self.getOptimizerD()
		self.optimizerG = self.getOptimizerG()
		self.one = torch.tensor(1, dtype=torch.float).cuda()
		self.mone = (self.one * -1).cuda()


	def optimizeParameters(self, input_batch, inputLabels=None):
		pass

	def getOptimizerG(self):
		g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
						  betas=[0.0, 0.999], lr=self.learningRate)

		return g_optimizer

	def getOptimizerD(self):
		return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
						  betas=[0, 0.999], lr=self.learningRate)

	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag

	def getSize(self):
		size = 2**(self.config.depth + 3)
		return (size, size)
