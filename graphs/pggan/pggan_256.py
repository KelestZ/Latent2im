from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch import autograd
from torch import nn
from .model_256 import *

class PGGAN():

	def __init__(self, lr, DIM=64):

		n_label = 1
		code_size = 512 - n_label

		self.netG = Generator(code_size, n_label).cuda()
		self.netD = Discriminator(n_label).cuda()
		self.g_running = Generator(code_size, n_label).cuda()
		self.g_running.train(False)

		self.learningRate = lr
		self.optimizerD = self.getOptimizerD()
		self.optimizerG = self.getOptimizerG()
		self.one = torch.tensor(1, dtype=torch.float).cuda()
		self.mone = (self.one * -1).cuda()


	def optimizeParameters(self, input_batch, inputLabels=None):
		pass

	def getOptimizerD(self):
		return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
						  betas=[0.5, 0.999], lr=self.learningRate)

	def getOptimizerG(self):
		return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
						  betas=[0.5, 0.999], lr=self.learningRate)

	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


	def getSize(self):
		size = 2**(self.config.depth + 3)
		return (size, size)
