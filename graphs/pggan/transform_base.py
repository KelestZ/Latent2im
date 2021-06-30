import torch, torchvision
import numpy as np
from . import constants, pggan_256 #pggan_128
from utils import image
import torch.nn as nn
from torch.autograd import Variable, grad
# from .gradient_penalty import gradient_penalty
import functools
import torch.nn.functional as F


class walk_embed(nn.Module):
	def __init__(self, dim_z, Nsliders):
		super(walk_embed, self).__init__()
		self.dim_z = dim_z
		self.Nsliders = Nsliders

		self.w_embed = nn.ParameterDict({
			'blondhair': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [6, 1, self.dim_z, Nsliders])).cuda()),
			'paleskin': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [6, 1, self.dim_z, Nsliders])).cuda()),
			'male': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [6, 1, self.dim_z, Nsliders])).cuda())})

	def forward(self, z, name, alpha, index_):
		z_new = z  #.cpu()
		for i in range(self.Nsliders):
			z_new = z_new + self.w_embed[name][index_, :, :, i]
			# al = torch.unsqueeze(alpha[:, i], axis=1)
			# z_new = z_new + al.cpu() * self.w_embed[name][index_, :, :, i]
		return z_new

# class walk_linear(nn.Module):
# 	def __init__(self, dim_z, Nsliders):
# 		super(walk_linear, self).__init__()
# 		self.dim_z = dim_z
# 		self.Nsliders = Nsliders
#
# 		self.w_embed = nn.ParameterDict({
# 			'blondhair': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders]))),
# 			'paleskin': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders]))),
# 			'male': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])))})
#
# 	def forward(self, z, name, alpha, index_):
# 		z_new = z.cpu()
# 		for i in range(self.Nsliders):
# 			al = torch.unsqueeze(alpha[:, i], axis=1)
# 			z_new = z_new + al.cpu() * self.w_embed[name][:, :, i]
# 		return z_new.cuda()
# class WalkLinearZ(nn.Module):
# 	def __init__(self, dim_z, step, Nsliders, attrList):
# 		super(WalkLinearZ, self).__init__()
# 		self.dim_z = dim_z
# 		self.step = step
# 		self.Nsliders = Nsliders
# 		self.w = nn.Parameter(
# 			torch.Tensor(np.random.normal(0.0, 0.02, [len(attrList), self.dim_z])))
#
# 	def forward(self, input, alpha, layers=None, name=None, index_=None):
# 		al = alpha.cuda()
# 		direction = torch.mm(al, self.w)  # B, C; C, 512
# 		out = input + direction
# 		return out

class WalkLinearZ(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkLinearZ, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		# Linear indenpendent
		# self.w = nn.Parameter(
		# 	torch.Tensor(np.random.normal(0.0, 0.02, [len(attrList), self.dim_z])))

		# Linear dependent
		self.linear = nn.Linear(self.dim_z, self.dim_z)

	def forward(self, input, alpha, layers=None, name=None, index_=None):
		al = alpha.cuda()
		# Linear dependent
		out = self.linear(input)
		direction = al * out / torch.norm(out, dim=1, keepdim=True) * 3
		out = input + direction
		return out


class WalkLinearZ_free(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkLinearZ_free, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		# Linear indenpendent
		self.w = nn.Parameter(
			torch.Tensor(np.random.normal(0.0, 0.02, [len(attrList), self.dim_z])))
		# Linear dependent
	def forward(self, input, alpha, layers=None, name=None, index_=None):
		al = alpha.cuda()
		# Linear dependent
		direction = al * input * self.w
		out = input + direction
		return out

class WalkMlpZ(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkMlpZ, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders
		# direction = np.zeros((1, len(attrList)))
		# direction[0, 0] = 1
		# self.direction = torch.Tensor(direction).cuda()
		self.embed = nn.Linear(len(attrList), self.dim_z//2)

		self.linear = nn.Sequential(*[nn.Linear(self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z)])

	def forward(self, input, alpha, layers=None, name=None, index_=None):
		# al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, C
		al = alpha.cuda()			# B, C
		# out = self.embed(al)		# B, dim_z
		# out2 = self.linear(torch.cat([out, input], 1))
		out2 = self.linear(input)

		# out2 = al * out2 / torch.norm(out2, dim=1, keepdim=True)
		# Single
		out2 = al * out2 / torch.norm(out2, dim=1, keepdim=True) * 3
		z_new = input + out2
		return z_new

class WalkMlpZ2(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkMlpZ2, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders
		# direction = np.zeros((1, len(attrList)))
		# direction[0, 0] = 1
		# self.direction = torch.Tensor(direction).cuda()

		self.linear = nn.Sequential(*[nn.Linear(self.dim_z, self.dim_z*2),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z*2, self.dim_z * 2),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z*2, self.dim_z)])
	# nn.Linear(self.dim_z2, self.dim_z * 2),
	# nn.LeakyReLU(0.2, True),
	def forward(self, input, alpha, layers=None, name=None, index_=None):
		# al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, C
		al = alpha.cuda()			# B, C
		# out = self.embed(al)		# B, dim_z
		# out2 = self.linear(torch.cat([out, input], 1))
		# out2 = al * out2 / torch.norm(out2, dim=1, keepdim=True)
		# Single

		out2 = self.linear(input)
		out2 = al * out2 / torch.norm(out2, dim=1, keepdim=True) * 3
		# out2 = al * out2
		z_new = input + out2
		return z_new

class WalkMlpZ3(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkMlpZ3, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders
		# direction = np.zeros((1, len(attrList)))
		# direction[0, 0] = 1
		# self.direction = torch.Tensor(direction).cuda()

		self.linear = nn.Sequential(*[nn.Linear(self.dim_z, self.dim_z*2),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z*2, self.dim_z * 2),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z*2, self.dim_z)])

	def forward(self, input, alpha, layers=None, name=None, index_=None):
		al = alpha.cuda()			# B, C
		out2 = self.linear(input)
		out2 = al * out2
		z_new = input + out2
		return z_new

class Normalization(nn.Module):
	def __init__(self):
		super(Normalization, self).__init__()
		mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
		std = torch.tensor([0.229, 0.224, 0.225]).cuda()

		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, img):
		return (img - self.mean) / self.std

class ContentLoss(nn.Module):
	def __init__(self):
		super(ContentLoss, self).__init__()

	def forward(self, org, shifted):
		self.loss = F.mse_loss(org.detach(), shifted)
		return self.loss


class TransformGraph():
	def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f,
				 trainEmbed, attrList, attrTable, layers, pgan_opts):
		assert (loss_type in ['l2', 'lpips']), 'unimplemented loss'

		# module inputs
		self.lr = lr
		self.useGPU = constants.useGPU
		# self.module = self.get_pgan_module()
		self.module = self.get_pgan_module_ganzoo()

		self.one = torch.tensor(1, dtype=torch.float).cuda()
		self.mone = (self.one * -1).cuda()
		self.regressor, self.reg_optmizer = self.get_reg_module()
		self.vgg19 = self.get_vgg_module()

		self.dim_z = constants.DIM_Z
		self.Nsliders = Nsliders = nsliders
		self.img_size = constants.resolution
		self.num_channels = constants.NUM_CHANNELS
		# self.CRITIC_ITERS = CRITIC_ITERS = constants.CRITIC_ITERS
		# self.OUTPUT_DIM = constants.OUTPUT_DIM
		self.BATCH_SIZE = constants.BATCH_SIZE
		self.LAMBDA = 0.1

		self.BCE_loss = nn.BCELoss()
		self.BCE_loss_logits = nn.BCEWithLogitsLoss()
		self.MSE_loss = nn.MSELoss()
		self.ContentLoss = ContentLoss()
		self.trainEmbed = trainEmbed

		# PGAN 256 - 6/128 - 5
		self.step = 6
		self.alpha = 0

		if not attrTable:
			self.attrTable = OrderedDict({
				'daylight': 1, 'night': 2, 'sunrisesunset': 3, 'sunny': 5,
				'clouds': 6, 'fog': 7, 'snow': 9, 'warm': 10, 'cold': 11,
				'beautiful': 13, 'flowers': 14, 'spring': 15, 'summer': 16,
				'autumn': 17, 'winter': 18, 'colorful': 20, 'dark': 24,
				'bright': 25, 'rain': 29, 'boring': 37, 'lush': 39})
		else:
			self.attrTable = attrTable
		self.attrList = attrList
		self.attrIdx = self.get_attr_idx()

		self.module.netG.eval()
		self.module.netD.eval()
		self.regressor.eval()
		self.reg_criterion = nn.MSELoss().cuda()

		print('LR 1e-4 for w, walk type: ', self.lr, walk_type)
		# walk pattern
		if walk_type == 'linear':
			print('Linear input not free')
			# self.walk = WalkLinearZ(self.dim_z, self.step, Nsliders, self.attrList).cuda()
			self.walk = WalkLinearZ_free(self.dim_z, self.step, Nsliders, self.attrList).cuda()

		else:
			# MLP
			print("MLP ")
			# self.walk = WalkMlpZ(self.dim_z, self.step, Nsliders, self.attrList).cuda()
			# self.walk = WalkMlpZ2(self.dim_z, self.step, Nsliders, self.attrList).cuda()
			self.walk = WalkMlpZ3(self.dim_z, self.step, Nsliders, self.attrList).cuda()

		self.optimizer = torch.optim.Adam(self.walk.parameters(), lr=self.lr, betas=(0.5, 0.99))

		# # set class vars
		self.Nsliders = Nsliders
		self.y = None
		self.z = None
		self.truncation = None

		"""
		# self.alpha = alpha
		# self.target = target
		# self.mask = mask
		# self.z_new = z_new
		# self.transformed_output = transformed_output
		# self.outputs_orig = outputs_orig
		# self.loss = loss
		# self.loss_lpips = loss_lpips
		# self.loss_l2_sample = loss_l2_sample
		# self.loss_lpips_sample = loss_lpips_sample
		# self.train_step = train_step
		
		"""

		self.walk_type = walk_type
		self.N_f = N_f  # NN num_steps
		self.eps = eps  # NN step_size
		self.BCE_loss = nn.BCELoss()

	def get_attr_idx(self):
		idxList = []
		for i in self.attrList:
			idxList.append(self.attrTable[i])
		return idxList

	def get_logits(self, inputs_dict, reshape=True):
		# print('check size: ', inputs_dict['z'].size())

		outputs_orig = self.module.netG(inputs_dict['z'])
		# if reshape == True:
		# Default: MNIST
		# outputs_orig = outputs_orig.view(-1, 1, 28, 28)
		# CelebA
		# outputs_orig = outputs_orig.view(-1, 3, constants.resolution, constants.resolution)
		downsampled = F.upsample(outputs_orig, size=(outputs_orig.size(2) // 2, outputs_orig.size(3) // 2), mode='bilinear')
		return downsampled#outputs_orig

	# def get_z_new(self, z, alpha):
	# 	if self.walk_type == 'linear' or self.walk_type == 'NNz':
	# 		z_new = z
	# 		for i in range(self.Nsliders):
	# 			# TODO: PROBLEM HERE
	# 			al = torch.unsqueeze(torch.Tensor(alpha[:, i]), axis=1)
	# 			z_new = (z_new + al * self.w[:, :, i]).cuda()
	# 	return z_new

	def get_z_new_tensor(self, z, alpha, name=None, trainEmbed=False, index_=None, layers=None):
		z_new = self.walk(z.squeeze(), alpha, name=name, index_=index_)
		return z_new

	def get_loss(self, feed_dict):
		# L2 loss
		target = feed_dict['target']
		mask = feed_dict['mask_out']
		logit = feed_dict['logit']
		diff = (logit - target) * mask
		return torch.sum(diff * diff) / torch.sum(mask)

	def get_edit_loss(self, feed_dict):
		# L2 loss
		target = feed_dict['target']
		mask = feed_dict['mask_out']
		logit = feed_dict['logit']
		diff = (logit - target) * mask
		return torch.sum(diff.pow(2)) / torch.sum(mask)

	def get_reg_preds(self, logit):
		preds = self.regressor(logit)[:, self.attrIdx]
		if len(preds.size()) == 1:
			preds = preds.unsqueeze(1)
		return preds

	def get_alphas(self, alpha_org, alpha_delta):
		# [N, C]
		alpha_target = torch.clamp(alpha_org + alpha_delta, min=0, max=1)
		alpha_delta_new = alpha_target - alpha_org
		# alpha_delta_new N, C
		# alpha_target N, C
		return alpha_target, alpha_delta_new

	def get_bce_loss(self, pred, y, eps=1e-12):
		loss = -(y * pred.clamp(min=eps).log() + (1 - y) * (1 - pred).clamp(min=eps).log()).mean()
		return loss

	def get_reg_loss(self, feed_dict):
		logit = feed_dict['logit']
		alpha_gt = feed_dict['alpha'].to(torch.double)
		# preds = self.regressor(logit)[:, self.attrTable[self.attrList[0]]]
		preds = self.regressor(logit)[:, self.attrIdx]
		preds = preds.unsqueeze(1).to(torch.double)
		# BCE
		loss = self.get_bce_loss(preds, alpha_gt)
		# MSE
		# loss = self.reg_criterion(preds, alpha_gt)
		return loss.mean()

	def get_content_loss(self, org_img, shifted_img):
		content_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']
		norm = Normalization().cuda()
		model = nn.Sequential(norm)
		# model = nn.Sequential()

		i = 0
		content_losses = []
		for layer in self.vgg19.children():
			if isinstance(layer, nn.Conv2d):
				i += 1
				name = 'conv_{}'.format(i)
			elif isinstance(layer, nn.ReLU):
				name = 'relu_{}'.format(i)
				layer = nn.ReLU(inplace=False)
			elif isinstance(layer, nn.MaxPool2d):
				name = 'pool_{}'.format(i)
			elif isinstance(layer, nn.BatchNorm2d):
				name = 'bn_{}'.format(i)
			else:
				raise RuntimeError('Unrecognized layer: {}'
								   .format(layer.__class__.__name__))
			model.add_module(name, layer)
			if name in content_layers:
				org_content = model(org_img).detach()
				shifted_content = model(shifted_img)
				content_loss = self.ContentLoss(org_content, shifted_content)
				# self.loss = F.mse_loss(org.detach(), shifted)
				content_losses.append(content_loss)
		return content_losses

	def optimizeParametersAll(self, feed_dict, trainEmbed, updateGAN,
							  no_content_loss=False, no_gan_loss=False):
		# FOR loaded PGAN
		# if updateGAN:
		# 	print('Update GAN')
		# 	# target = feed_dict['target']
		# 	mask = feed_dict['mask_out']
		# 	logit = feed_dict['logit']
		# 	x_real = feed_dict['real_target']
		#
		# 	y_real = Variable(torch.ones(logit.size()[0]).cuda())
		# 	y_fake = Variable(torch.zeros(logit.size()[0]).cuda())
		#
		# 	# Update D
		# 	self.module.optimizerD.zero_grad()
		#
		# 	# D real
		# 	D_real_result = self.module.netD(x_real).squeeze()
		# 	D_real_result = D_real_result.mean() - 0.001 * (D_real_result ** 2).mean()
		# 	D_real_result.backward(self.module.mone, retain_graph=True)
		#
		# 	# D fake
		# 	D_fake_result = self.module.netD(logit).squeeze()
		# 	D_fake_result = D_fake_result.mean()
		# 	D_fake_result.backward(self.module.one, retain_graph=True)
		#
		# 	# TRAIN WITH GRADIENT PENALTY
		# 	# gp = gradient_penalty(functools.partial(self.module.netD), x_real, logit,
		# 	# 					  gp_mode='1-gp',
		# 	# 					  sample_mode='line')
		# 	# gradient_penalty = calc_gradient_penalty(self.module.netD, x_real.data, logit.data, self.BATCH_SIZE)
		#
		# 	eps = torch.rand(constants.BATCH_SIZE, 1, 1, 1).cuda()
		# 	x_hat = eps * x_real.data + (1 - eps) * logit.data
		# 	x_hat = Variable(x_hat, requires_grad=True)
		# 	hat_predict = self.module.netD(x_hat)
		# 	grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
		# 	grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
		# 	grad_penalty = 10 * grad_penalty
		# 	grad_penalty.backward(retain_graph=True)
		#
		# 	# grad_loss_val = grad_penalty.data
		# 	# disc_loss_val = (real_predict - fake_predict).data
		# 	self.module.optimizerD.step()
		#
		# 	# Update G
		# 	self.module.optimizerG.zero_grad()
		# 	new_logit = self.get_logits(feed_dict)
		# 	feed_dict['logit'] = new_logit
		#
		# 	D_fake_result = self.module.netD(new_logit).squeeze()
		# 	G_train_loss = self.BCE_loss_logits(D_fake_result, y_real)
		#
		# 	Edit_loss = self.get_edit_loss(feed_dict)
		# 	G_train_loss += self.LAMBDA * Edit_loss
		# 	G_train_loss.backward(retain_graph=True)
		# 	self.module.optimizerG.step()
		# 	self.module.accumulate(self.module.g_running, self.module.netG)

		# Update w
		#if trainEmbed:
		#	print('Update w_embed')

		self.optimizer.zero_grad()
		# D loss
		logit = feed_dict['logit']
		# D_fake_result, _ = self.module.netD(logit)  # B, 1
		D_fake_result = self.module.netD(F.upsample(logit, size=(int(logit.size(2)*2), int(logit.size(3)*2)), mode='bilinear'))  # B, 1

		y_real = Variable(torch.ones_like(D_fake_result).cuda())
		gan_loss = self.BCE_loss_logits(D_fake_result, y_real)

		# Content loss
		content_loss_list = self.get_content_loss(feed_dict['org'],
												  feed_dict['logit'])
		content_losses = 0
		for i in range(len(content_loss_list)):
			content_losses += content_loss_list[i]
		content_losses = content_losses / len(content_loss_list)

		# Regression loss
		reg_loss = self.get_reg_loss(feed_dict)

		if no_content_loss or  no_gan_loss:
			loss = reg_loss
		else:
			loss = 10 * reg_loss

		if not no_content_loss:
			loss += 0.05 * content_losses  # 0.05 * content_losses
		if not no_gan_loss:
			loss += 0.05 * gan_loss
		loss.backward()
		# for name, parms in self.walk.linear.named_parameters():
		# 	print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
		# 		  ' -->grad_value:', parms.grad)
		self.optimizer.step()
		return loss

	def save_multi_models(self, save_path_w, save_path_gan,
						  trainEmbed=False, updateGAN=False,
						  single_transform_name=None):
		print('Save W and GAN in %s and %s' % (save_path_w, save_path_gan))
		if updateGAN == True:
			print('Save GAN')
			torch.save(self.module, save_path_gan)
		torch.save(self.walk, save_path_w + '_walk_module.ckpt')

	def load_multi_models(self, save_path_w, save_path_gan,
						  trainEmbed=False, updateGAN=False,
						  single_transform_name=None):
		if updateGAN:
			print('Load GAN in %s' % save_path_gan)
			self.module = torch.load(save_path_gan)

		print('Load w in %s' % save_path_w)
		self.walk = torch.load(save_path_w)

	def get_reg_module(self):
		#####
		model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False)
		model.fc = torch.nn.Linear(2048, 40)
		model = model.cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
		#####
		# base_dir = '/home/peiye/ImageEditing/scene_regressor/checkpoint_256/'
		# ckpt = torch.load(base_dir + '500_dict.model')

		base_dir = '/shared/rsaas/zpy/celeba_reg/checkpoint/'
		ckpt = torch.load(base_dir + '108_dict.model')
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optm'])
		return model, optimizer


	def get_vgg_module(self):
		#####
		import torchvision.models as models
		vgg19 = models.vgg19(pretrained=True).features.cuda().eval()
		return vgg19

	def get_pgan_module_ganzoo(self):
		print('Loading PGGAN module from ganzoo')
		# 256
		# module = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
		# 						pretrained=True,
		# 						model_name='celebAHQ-256',
		# 						useGPU=self.useGPU)

		# 512
		module = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
							   'PGAN', model_name='celebAHQ-512',
							   pretrained=True, useGPU=True)
		return module

	def get_pgan_module(self):
		print('Loading PGGAN module')
		# generator = model.Generator(511, 1).cuda().train(False)
		# module = pggan_128.PGGAN(lr=self.lr)
		# load_pretrain
		# base_dir = '/home/peiye/ImageEditing/progressive-gan-pytorch/checkpoint/'
		# ckpt = torch.load(base_dir + '510000_dict.model')
		# base_dir = '/home/peiye/ImageEditing/vision_16_pgan/checkpoint/'

		module = pggan_256.PGGAN(lr=self.lr)

		base_dir = '/home/peiye/ImageEditing/pgan_scene/pgan256/checkpoint/'
		# ckpt = torch.load(base_dir + '550000_dict.model')
		ckpt = torch.load(base_dir + '280000_dict.model')
		print('Start loading PGGAN_scene 256 module in %s' % (base_dir))

		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in ckpt['G'].items():
			name = k[7:]
			new_state_dict[name] = v
		module.netG.load_state_dict(new_state_dict)

		new_state_dict = OrderedDict()
		for k, v in ckpt['D'].items():
			name = k[7:]
			new_state_dict[name] = v
		module.netD.load_state_dict(new_state_dict)
		print('Finish loading the pretrained model')
		return module

	def clip_ims(self, ims):
		return np.uint8(np.clip(((ims + 1) / 2.0) * 256, 0, 255))

	# FOR WGANGP
	# def clip_ims(self, ims):
	# 	return np.uint8(np.clip(ims * 255, 0, 255))

	def apply_alpha(self, graph_inputs, alpha_to_graph,
					layers=None, name=None,
					trainEmbed=False, index_=None,
					given_w=None):
		zs_batch = graph_inputs['z']  # tensor.cuda() # [Batch, DIM_Z]
		feed_dict = {'z': zs_batch}
		out_zs = self.get_logits(feed_dict)
		alpha_to_graph = torch.tensor(alpha_to_graph).float().cuda()
		alpha_org = self.get_reg_preds(out_zs)  # [B, C]
		alpha_delta = torch.zeros_like(alpha_org).cuda()  # alpha_org - alpha_org
		######
		# BUG:
		# alpha_delta
		alpha_delta = alpha_to_graph - alpha_org

		# if index_ != None:
		# 	# alpha_to_graph [B, 1]
		# 	# alpha_delta [B, attrList]
		# 	# for j in range(len(self.attrIdx)):
		# 	if len(self.attrIdx) == len(self.attrTable):
		# 		alpha_delta[:, index_] = torch.Tensor(alpha_to_graph[:, 0]).cuda() - alpha_org[:, index_]
		# 	# print('alpha_delta[:, index_]: ', alpha_delta[:, index_])
		# 	else:
		# 		i = self.attrIdx.index(index_)
		# 		# print('Here: ', index_, i, alpha_to_graph.shape, alpha_org.size())
		# 		alpha_delta[:, i] = torch.Tensor(alpha_to_graph[:, 0]).cuda() - alpha_org[:, i]
		#
		# print('alpha_to_graph.size(): ', alpha_to_graph.ssize(), alpha_org.size())
		# print('alpha_delta: ', alpha_delta,  alpha_delta.size(), self.attrIdx)

		z_new = self.get_z_new_tensor(zs_batch, alpha_delta, name, trainEmbed=trainEmbed, index_=index_)
		best_inputs = {'z': z_new}

		best_im_out = self.get_logits(best_inputs)
		return best_im_out, alpha_org

	def L2_loss(self, img1, img2, mask):
		return np.sum(np.square((img1 - img2) * mask), (1, 2, 3))

	def vis_image_batch_alphas(self, graph_inputs, filename,
							   alphas_to_graph, alphas_to_target,
							   batch_start, name=None, wgt=False, wmask=False,
							   trainEmbed=False, computeL2=True):

		zs_batch = graph_inputs['z']  # numpy
		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []

		# index_ = [i in range(len(alphas_to_graph))]
		L2_loss = {}
		print('len(alphas_to_graph)', len(alphas_to_graph))

		for index_, (ag, at) in enumerator(zip(alphas_to_graph, alphas_to_target)):
			print('Index: ', index)

			input_test = {'z': torch.Tensor(zs_batch).cuda()}
			out_input_test = self.get_logits(input_test)
			out_input_test = out_input_test.detach().cpu().numpy()  # on Cuda
			target_fn, mask_out = self.get_target_np(out_input_test, at)

			best_im_out = self.apply_alpha(input_test, ag, name, index_).detach().cpu().numpy()

			L2_loss[at] = self.L2_loss(target_fn, best_im_out, mask_out)
			ims_target.append(target_fn)
			ims_transformed.append(best_im_out)
			ims_mask.append(mask_out)
		if computeL2:
			## Compute L2 loss for drawing the plots
			return L2_loss

		######### ######### ######### ######### #########
		print('wgt: ', wgt)
		for ii in range(zs_batch.shape[0]):
			arr_gt = np.stack([x[ii, :, :, :] for x in ims_target], axis=0)

			if wmask:
				arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
										  in zip(ims_transformed, ims_mask)], axis=0)
			else:
				arr_transform = np.stack([x[ii, :, :, :] for x in
										  ims_transformed], axis=0)
			arr_gt = self.clip_ims(arr_gt)
			arr_transform = self.clip_ims(arr_transform)
			if wgt:
				ims = np.concatenate((arr_gt, arr_transform), axis=0)
			else:
				ims = arr_transform
			filename = filename_base + '_sample{}'.format(ii + batch_start)
			if wgt:
				filename += '_wgt'
			if wmask:
				filename += '_wmask'
			# (7, 3, 64, 64)
			if ims.shape[1] == self.num_channels:
				# N C W H -> N W H C
				ims = np.transpose(ims, [0, 2, 3, 1])
			# ims = np.squeeze(ims)
			# print('ims.shape: ', ims.shape)
			image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

	def vis_multi_image_batch_alphas(self, graph_inputs, filename,
									 alphas_to_graph, alphas_to_target,
									 batch_start,
									 layers=None,
									 name=None, wgt=False, wmask=False,
									 trainEmbed=False, computeL2=False,
									 given_w=None, index_=None):
		# TODO:
		# CHANGE!!
		zs_batch = graph_inputs['z']  # numpy

		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []
		L2_loss = {}
		index_ = 0
		for ag, at in zip(alphas_to_graph, alphas_to_target):
			input_test = {'z': torch.Tensor(zs_batch).cuda()}

			best_im_out, alpha_org = self.apply_alpha(input_test, ag, name=name, layers=layers,
										   trainEmbed=trainEmbed, given_w=given_w, index_=index_)
			# best_im_out = F.interpolate(best_im_out, size=256)
			best_im_out = best_im_out.detach().cpu().numpy()
			best_im_out = np.uint8(np.clip(((best_im_out + 1) / 2.0) * 255, 0, 255))
			ims_transformed.append(best_im_out)


		for ii in range(zs_batch.shape[0]):
			if wmask:
				arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
										  in zip(ims_transformed, ims_mask)], axis=0)
			else:
				arr_transform = np.stack([x[ii, :, :, :] for x in
										  ims_transformed], axis=0)
			ims = arr_transform
			filename = filename_base + '_sample{}'.format(ii + batch_start)
			if wgt:
				filename += '_wgt'
			if wmask:
				filename += '_wmask'
			# (7, 3, 64, 64)
			if ims.shape[1] == 1 or ims.shape[1] == 3:
				# N C W H -> N W H C
				ims = np.transpose(ims, [0, 2, 3, 1])
			# ims = np.squeeze(ims)

			a_org = alpha_org[ii, ]

			image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

	def vis_image_batch(self, graph_inputs, filename,
						batch_start, wgt=False, wmask=False, num_panels=7):
		raise NotImplementedError('Subclass should implement vis_image_batch')


class BboxTransform(TransformGraph):
	def __init__(self, *args, **kwargs):
		TransformGraph.__init__(self, *args, **kwargs)

	def get_distribution_statistic(self, img, channel=None):
		raise NotImplementedError('Subclass should implement get_distribution_statistic')

	# def get_category_list(self):
	#	return get_coco_imagenet_categories()

	def distribution_data_per_category(self, num_categories, num_samples,
									   output_path, channel=None):
		raise NotImplementedError('Coming soon')

	def distribution_model_per_category(self, num_categories, num_samples,
										a, output_path, channel=None):
		raise NotImplementedError('Coming soon')

	def get_distributions_per_category(self, num_categories, num_samples,
									   output_path, palpha, nalpha,
									   channel=None):
		raise NotImplementedError('Coming soon')

	def get_distributions_all_categories(self, num_samples, output_path,
										 channel=None):
		raise NotImplementedError('Coming soon')


class PixelTransform(TransformGraph):
	def __init__(self, *args, **kwargs):
		TransformGraph.__init__(self, *args, **kwargs)

	def get_distribution_statistic(self, img, channel=None):
		raise NotImplementedError('Subclass should implement get_distribution_statistic')

	def get_distribution(self, num_samples, channel=None):
		random_seed = 0
		rnd = np.random.RandomState(random_seed)
		inputs = graph_input(self, num_samples, seed=random_seed)
		batch_size = constants.BATCH_SIZE
		model_samples = []
		for a in self.test_alphas():
			distribution = []
			start = time.time()
			print("Computing attribute statistic for alpha={:0.2f}".format(a))
			for batch_num, batch_start in enumerate(range(0, num_samples,
														  batch_size)):
				s = slice(batch_start, min(num_samples, batch_start + batch_size))
				inputs_batch = util.batch_input(inputs, s)
				zs_batch = inputs_batch[self.z]
				a_graph = self.scale_test_alpha_for_graph(a, zs_batch, channel)
				ims = self.clip_ims(self.apply_alpha(inputs_batch, a_graph))
				for img in ims:
					img_stat = self.get_distribution_statistic(img, channel)
					distribution.extend(img_stat)
			end = time.time()
			print("Sampled {} images in {:0.2f} min".format(num_samples, (end - start) / 60))
			model_samples.append(distribution)

		model_samples = np.array(model_samples)
		return model_samples


"""
	def load_multi_models(self, save_path_w, save_path_gan, trainEmbed=False, updateGAN=False,
						  single_transform_name=None):
		# Load GAN
		print('Load GAN in %s' % save_path_gan)
		# self.module.load(save_path_gan)
		if updateGAN:
			print('Load GAN')
			self.module = torch.load(save_path_gan)

		print('Load w in %s' % save_path_w)
		try:
			self.walk = torch.load(save_path_w)
		except:
			for name in self.walk.w_embed.keys():
				new_w_path = save_path_w + '_' + name + '.npy'
				print('Load W of %s' % name)
				print('Before w: ', self.walk.w_embed[name].size())
				self.walk.w_embed[name] = torch.nn.Parameter(torch.Tensor(np.load(new_w_path)).cuda())
				print('After w: ', self.walk.w_embed[name].size())
		# if trainEmbed:
		# 	# Load w
		# 	if single_transform_name:
		# 		print('Load %s only ' % single_transform_name)
		# 		new_w_path = save_path_w + '_' + single_transform_name + '.npy'
		# 		print('Before w: ', self.w_embed[single_transform_name][0, :5, 0])
		# 		self.walk.w_embed[single_transform_name] = torch.Tensor(np.load(new_w_path))
		# 		print('After w: ', self.w_embed[single_transform_name][0, :5, 0])
		# 		return
		#
		# 	print('Load W for embedding')
		# 	for name in self.w_embed.keys():
		# 		new_w_path = save_path_w + '_' + name + '.npy'
		# 		print('Load W of %s' % name)
		# 		print('Before w: ', self.w_embed[name][0, :5, 0])
		# 		self.walk.w_embed[name] = torch.Tensor(np.load(new_w_path))
		# 		print('After w: ', self.w_embed[name][0, :5, 0])
		# else:
		# 	print('Load continuous Ws ')
		# 	for name in self.ws.keys():
		# 		new_w_path = save_path_w + '_' + name + '.npy'
		# 		print('Load W of %s' % name)
		# 		print('Before w: ', self.ws[name][0, :5, 0])
		# 		self.ws[name] = torch.Tensor(np.load(new_w_path))
		# 		print('After w: ', self.ws[name][0, :5, 0])
	def load_model(self, save_path_w, save_path_gan):
		# Load w
		print('Load W in %s' % save_path_w)
		print('Before w: ', self.w[0, :5, 0])
		self.w = torch.Tensor(np.load(save_path_w))
		print('After w: ', self.w[0, :5, 0])
		# Load GAN
		print('Load GAN in %s' % save_path_gan)
		# self.module.load(save_path_gan)
		self.module = torch.load(save_path_gan)
"""
"""
	def save_model(self, save_path_w, save_path_gan):
		print('Save W and GAN in %s and %s' % (save_path_w, save_path_gan))
		torch.save(self.module, save_path_gan)
		# self.module.save(save_path_gan)
		np.save(save_path_w, self.w.detach().cpu().numpy())

	def save_multi_models(self, save_path_w, save_path_gan, trainEmbed=False, updateGAN=True,
						  single_transform_name=None):
		print('Save W and GAN in %s and %s' % (save_path_w, save_path_gan))
		if updateGAN == True:
			print('Save GAN')
			torch.save(self.module, save_path_gan)
		# self.module.save(save_path_gan)

		torch.save(self.walk, save_path_w + '_walk_module.ckpt')

		if trainEmbed:
			if single_transform_name:
				print('Save %s only ' % single_transform_name)
				cur_path_w = save_path_w + '_' + single_transform_name
				np.save(cur_path_w, self.walk.w_embe[single_transform_name].detach().cpu().numpy())
				return
			print('Save embed W')
			for i, cur_w in self.walk.w_embed.items():
				cur_path_w = save_path_w + '_' + i
				np.save(cur_path_w, cur_w.detach().cpu().numpy())
			return
		else:
			print('Save ws')
			for i, cur_w in self.walk.w_embed.items():
				cur_path_w = save_path_w + '_' + i
				np.save(cur_path_w, cur_w.detach().cpu().numpy())

"""
'''
def optimizeParameters(self, feed_dict, name=None, updateGAN=True, trainEmbed=False):
	"""
	Used to update single operation. Next one is compatable (better)
	"""
	#target = feed_dict['target']
	mask = feed_dict['mask_out']
	logit = feed_dict['logit']
	x_real = feed_dict['real_target']
	# Train discreted W only
	if updateGAN == False and trainEmbed == True:
		self.optimizers_embed[name].zero_grad()
		Edit_loss = self.get_edit_loss(feed_dict)
		Edit_loss.backward(retain_graph=True)
		self.optimizers_embed[name].step()
		return Edit_loss

	elif updateGAN == False and trainEmbed == False:
		raise('ERROR')

	## ------- updateGAN == True, (ignore trainEmbed) ------- ##
	# Update D
	for iter_d in range(self.CRITIC_ITERS):
		self.module.optimizerD.zero_grad()
		logit = self.get_logits(feed_dict, reshape=False)
		#  train with real
		# x_resized = x_real.view(-1, self.OUTPUT_DIM)
		D_real = self.module.netD(x_real)
		D_real = D_real.mean()
		D_real.backward(self.module.mone, retain_graph=True)

		D_fake = self.module.netD(logit)
		D_fake = D_fake.mean()
		D_fake.backward(self.module.one, retain_graph=True)

		# train with gradient penalty
		gp = gradient_penalty(functools.partial(self.module.netD), x_real, logit,
											gp_mode='1-gp',
											sample_mode='line')

		# gradient_penalty = calc_gradient_penalty(self.module.netD, x_real.data, logit.data, self.BATCH_SIZE)
		gp.backward(retain_graph=True)

		# Wasserstein_D = D_real - D_fake
		self.module.optimizerD.step()

	# Update G
	self.module.optimizerG.zero_grad()

	new_logit = self.get_logits(feed_dict)
	feed_dict['logit'] = new_logit
	# G_train_loss = self.module.netD(new_logit.view(-1, self.OUTPUT_DIM)).mean()
	G_train_loss = self.module.netD(new_logit).mean()
	Edit_loss = - self.get_edit_loss(feed_dict)
	G_train_loss += Edit_loss
	# print('Edit_loss1 and G loss: ', Edit_loss, G_train_loss)

	G_train_loss.backward(self.module.mone, retain_graph=True)
	self.module.optimizerG.step()
	# Update w
	if name:
		if trainEmbed:
			self.optimizers_embed[name].zero_grad()
			new_logit = self.get_logits(feed_dict)
			feed_dict['logit'] = new_logit
			Edit_loss = self.get_edit_loss(feed_dict)
			# print('Edit_loss2: ', Edit_loss)
			Edit_loss.backward(retain_graph=True)
			self.optimizers_embed[name].step()
		else:
			self.optimizers[name].zero_grad()
			new_logit = self.get_logits(feed_dict)
			feed_dict['logit'] = new_logit
			Edit_loss = self.get_edit_loss(feed_dict)
			# print('Edit_loss2: ', Edit_loss)
			Edit_loss.backward(retain_graph=True)
			self.optimizers[name].step()

	else:
		self.optimizer.zero_grad()
		new_logit = self.get_logits(feed_dict)
		feed_dict['logit'] = new_logit
		Edit_loss = self.get_edit_loss(feed_dict)
		# print('Edit_loss2: ', Edit_loss)
		Edit_loss.backward(retain_graph=True)
		self.optimizer.step()

	return Edit_loss

def optimizeParametersAll(self, feed_dict):

	x_real = feed_dict['real_target']

	# Update D
	for iter_d in range(self.CRITIC_ITERS):
		self.module.optimizerD.zero_grad()
		logit = self.get_logits(feed_dict, reshape=False)
		#  train with real
		# x_resized = x_real.view(-1, self.OUTPUT_DIM)
		# D_real = self.module.netD(x_resized)
		D_real = self.module.netD(x_real)
		D_real = D_real.mean()
		D_real.backward(self.module.mone, retain_graph=True)

		D_fake = self.module.netD(logit)
		D_fake = D_fake.mean()
		D_fake.backward(self.module.one, retain_graph=True)

		# train with gradient penalty
		# gradient_penalty = calc_gradient_penalty(self.module.netD, x_resized.data, logit.data, self.BATCH_SIZE)
		gp = gradient_penalty(functools.partial(self.module.netD), x_real, logit,
											gp_mode='1-gp',
											sample_mode='line')

		gp.backward(retain_graph=True)
		# Wasserstein_D = D_real - D_fake
		self.module.optimizerD.step()

	# Update G
	self.module.optimizerG.zero_grad()
	new_logit = self.get_logits(feed_dict)
	feed_dict['logit'] = new_logit
	# G_train_loss = self.module.netD(new_logit.view(-1, self.OUTPUT_DIM)).mean()

	G_train_loss = self.module.netD(new_logit).mean()
	Edit_loss = - self.get_edit_loss(feed_dict)
	G_train_loss += Edit_loss
	G_train_loss.backward(self.module.mone, retain_graph=True)
	self.module.optimizerG.step()

	# Update w
	for name in self.optimizers.keys():
		self.optimizers[name].zero_grad()
		new_logit = self.get_logits(feed_dict)
		feed_dict['logit'] = new_logit
		Edit_loss = self.get_edit_loss(feed_dict)
		# print('Edit_loss2: ', Edit_loss)

		Edit_loss.backward(retain_graph=True)
		self.optimizers[name].step()

	# else:
	# 	self.optimizer.zero_grad()
	# 	new_logit = self.get_logits(feed_dict)
	# 	feed_dict['logit'] = new_logit
	# 	Edit_loss = self.get_edit_loss(feed_dict)
	# 	# print('Edit_loss2: ', Edit_loss)
	# 	Edit_loss.backward(retain_graph=True)
	# 	self.optimizer.step()

	return Edit_loss


def optimizeParametersAll(self, feed_dict, trainEmbed, updateGAN):
	if updateGAN:
		print('Update GAN')
		# target = feed_dict['target']
		mask = feed_dict['mask_out']
		logit = feed_dict['logit']
		x_real = feed_dict['real_target']

		y_real = Variable(torch.ones(logit.size()[0]).cuda())
		y_fake = Variable(torch.zeros(logit.size()[0]).cuda())

		# Update D
		self.module.optimizerD.zero_grad()
		D_real_result = self.module.netD(x_real).squeeze()
		# print(D_real_result)

		D_real_loss = self.BCE_loss_logits(D_real_result, y_real)
		D_fake_result = self.module.netD(logit).squeeze()
		D_fake_loss = self.BCE_loss_logits(D_fake_result, y_fake)
		D_train_loss = D_real_loss + D_fake_loss
		# print('D_train_loss: ', D_train_loss)
		D_train_loss.backward(retain_graph=True)
		self.module.optimizerD.step()

		# Update G
		self.module.optimizerG.zero_grad()
		new_logit = self.get_logits(feed_dict)
		feed_dict['logit'] = new_logit

		D_fake_result = self.module.netD(new_logit).squeeze()
		G_train_loss = self.BCE_loss_logits(D_fake_result, y_real)
		Edit_loss = self.get_edit_loss(feed_dict)
		G_train_loss += self.LAMBDA * Edit_loss
		# print('G_train_loss: ', G_train_loss)
		G_train_loss.backward(retain_graph=True)
		self.module.optimizerG.step()

	# Update w
	if trainEmbed:
		for name in self.optimizers_embed.keys():
			self.optimizers_embed[name].zero_grad()
			new_logit = self.get_logits(feed_dict)
			feed_dict['logit'] = new_logit
			Edit_loss = self.get_edit_loss(feed_dict)
			# print('Edit_loss2: ', Edit_loss)

			Edit_loss.backward(retain_graph=True)
			self.optimizers_embed[name].step()
	else:
		raise('ERROR')
	return Edit_loss
'''
