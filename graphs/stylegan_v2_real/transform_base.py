import os
import torch, torchvision
import numpy as np
from utils import image
import torch.nn as nn
from torch.autograd import Variable, grad
# from .gradient_penalty import gradient_penalty
from torch import nn, optim
import functools
import torch.nn.functional as F
from . import constants
from . import stylegan2 as stylegan

from torchvision import utils as ut
import json
import torchvision.models as models
from collections import OrderedDict
from easydict import EasyDict as edict


class WalkEmbed(nn.Module):
	def __init__(self, dim_z, Nsliders, attrList):
		super(WalkEmbed, self).__init__()
		"""
		This is a test module where an embedding bank strategy is used for discrete attribute values.
		This module is not used in the end.
		"""
		self.dim_z = dim_z
		self.Nsliders = Nsliders
		self.w = nn.ParameterDict()
		for i in attrList:
			self.w.update(
				{i: nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [6, 1, self.dim_z, Nsliders])).cuda())})

	def forward(self, z, name, alpha, index_):
		z_new = z  # .cpu()
		for i in range(self.Nsliders):
			z_new = z_new + self.w[name][index_, :, :, i]
		# al = torch.unsqueeze(alpha[:, i], axis=1)
		# z_new = z_new + al.cpu() * self.w_embed[name][index_, :, :, i]
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


class WalkLinear(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkLinear, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.w = nn.ParameterDict()
		for i in attrList:
			self.w.update(
				{i: nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])).cuda())})

	def forward(self, z, name, alpha, index_):
		z_new = z.cpu()
		for i in range(self.Nsliders):
			al = torch.unsqueeze(alpha[:, i], axis=1)
			z_new = z_new + al.cpu() * self.w[name][:, :, i].cpu()
		return z_new.cuda()


class WalkMlpMultiZ(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkMlpMultiZ, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders
		direction = np.zeros((1, 10))
		direction[0, 0] = 1
		self.direction = torch.Tensor(direction).cuda()
		self.embed = nn.Linear(10, self.dim_z)
		self.linear = nn.Sequential(*[nn.Linear(self.dim_z + self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z)])

	def forward(self, input, name, alpha, index_, layers=None):
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, 1

		out = self.embed(self.direction.repeat(al.size(0), 1))
		print('MLP z')
		out2 = self.linear(torch.cat([out, input], 1))
		out2 = al * out2 / torch.norm(out2, dim=1, keepdim=True)
		z_new = input + out2

		return z_new

#
# class WalkLinearSingleW(nn.Module):
# 	def __init__(self, dim_z, step, Nsliders, attrList):
# 		super(WalkLinearSingleW, self).__init__()
# 		self.dim_z = dim_z
# 		self.step = step
# 		self.Nsliders = Nsliders
#
# 		self.w = nn.Parameter(
# 			torch.Tensor(np.random.normal(0.0, 0.02, [len(attrList), self.dim_z])))
#
# 	def forward(self, input, alpha, layers=None, name=None, index_=None):
# 		w_transformed = []
# 		al = alpha.cuda()
# 		# alpha, al in [B, C]
#
# 		for i in range(len(input)):
# 			if layers == None or i in layers:
# 				direction = torch.mm(al, self.w)		# B, C; C, 512
# 				out = input[i] + direction
# 			else:
# 				out = input[i]
# 			w_transformed.append(out)
# 		return w_transformed
#

# Input Independent W
class WalkLinearMultiW(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkLinearMultiW, self).__init__()
		# self.Nsliders = Nsliders
		self.dim_z = dim_z
		self.step = step
		# A, 14, 512
		self.w = nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [len(attrList), (self.step + 1) * 2, self.dim_z])))
		# self.w = nn.Parameter(
		# 	torch.Tensor(np.random.normal(0.0, 0.02, [len(attrList), self.dim_z])))

	def forward(self, input, alpha, layers=None,
				name=None, index_=None):
		w_transformed = []
		# al = torch.unsqueeze(alpha[:, :], axis=-1).cuda()
		al = alpha.cuda()
		# alpha, al in [B, C]

		for i in range(len(input)):
			if layers == None or i in layers:
				direction = torch.mm(al, self.w[:, i, :])		# B, C; C, 512
				out = input[i] + direction
			else:
				out = input[i]
			w_transformed.append(out)
		return w_transformed

# d = MLP for StyleGAN2
class WalkMlpMultiW(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkMlpMultiW, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.linear = nn.Sequential(*[nn.Linear(self.dim_z, 2 * self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(2 * self.dim_z, 2* self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(2*self.dim_z, self.dim_z)])

	def forward(self, input, alpha, layers=None, name=None, index_=None):

		w_transformed = []
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, 1

		if layers == None:
			for i in range(len(input)):
				out2 = self.linear(input[i])
				out2 = out2 # / torch.norm(out2, dim=1, keepdim=True) * 3

				w_new = input[i] + al * out2
				# w_new = torch.clamp(w_new, min=-1, max=2)
				w_transformed.append(w_new)
			return w_transformed

		for i in range(len(input)):
			if i in layers:
				out2 = self.linear(input[i], 1)
				# out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + al * out2
			else:
				w_new = input[i]
			w_transformed.append(w_new)
		return w_transformed

#  d = MLP for Progressive GAN
class WalkNonLinearW(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(WalkNonLinearW, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.embed = nn.Linear(10, self.dim_z // 2)
		self.linear = nn.Sequential(*[nn.Linear(self.dim_z // 2 + self.dim_z, 2 * self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(2 * self.dim_z, self.dim_z)])

	def forward(self, input, name, alpha, index_, layers=None):
		w_transformed = []
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, 1
		out = self.embed(al.repeat(1, 10))
		if layers == None:
			for i in range(len(input)):
				# print('Max min before: ', input[i].max(), input[i].min())

				out2 = self.linear(torch.cat([out, input[i]], 1))
				out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + out2
				# w_new = torch.clamp(w_new, min=-1, max=2)
				w_transformed.append(w_new)

			return w_transformed

		for i in range(len(input)):
			if i in layers:
				out2 = self.linear(torch.cat([out, input[i]], 1))
				# out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + out2
			else:
				w_new = input[i]
			w_transformed.append(w_new)
		return w_transformed


class TransformGraph():
	def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f,
				 trainEmbed, attrList, attrTable, layers, stylegan_opts):
		assert (loss_type in ['l2', 'lpips']), 'unimplemented loss'

		# module inputs
		self.lr = lr
		self.useGPU = constants.useGPU
		self.module = self.get_stylegan2_module()

		self.one = torch.tensor(1, dtype=torch.float).cuda()
		self.mone = (self.one * -1).cuda()
		self.regressor, self.reg_optmizer = self.get_reg_module()
		self.vgg19 = self.get_vgg_module()

		self.attrTable = attrTable
		self.attrList = attrList
		self.attrIdx = self.get_attr_idx()

		self.module.netG.eval()
		self.module.netD.eval()
		self.regressor.eval()


		self.dim_z = constants.DIM_Z
		self.Nsliders = Nsliders = nsliders
		self.img_size = constants.resolution
		self.num_channels = constants.NUM_CHANNELS
		self.BATCH_SIZE = constants.BATCH_SIZE
		self.LAMBDA = 0.05

		self.BCE_loss = nn.BCELoss()
		self.BCE_loss_logits = nn.BCEWithLogitsLoss()
		self.MSE_loss = nn.MSELoss()
		self.ContentLoss = ContentLoss()

		self.trainEmbed = trainEmbed

		# StyleAN 256
		self.step = 6
		self.alpha = 1
		self.stylegan_opts = stylegan_opts
		self.layers = layers

		# TODO: Hard code
		self.is_mlp = False
		# self.is_single = False

		# walk pattern
		print('walk_type and tylegan_opts.latent: ', walk_type, stylegan_opts.latent)

		if walk_type == 'linear':
			if self.trainEmbed == True:
				"""
				An unused setting in the paper. Could ignoire this.
				"""
				print('Walk in non-linear embed')
				self.walk = WalkEmbed(self.dim_z, Nsliders, self.attrList)
			else:
				if stylegan_opts.latent == 'z':
					if self.is_mlp:
						self.walk = WalkMlpMultiZ(self.dim_z, self.step, Nsliders, self.attrList).cuda()
					else:
						raise NotImplementedError('Not implemented setting of linear transformation for z')

				elif stylegan_opts.latent == 'w':
					if self.is_mlp:
						"""
						An unused settting in the paper
						"""
						self.walk = WalkMlpMultiW(self.dim_z, self.step,
												  Nsliders, self.attrList).cuda()
					else:
						self.walk = WalkLinearMultiW(self.dim_z, self.step,
													 Nsliders, self.attrList).cuda()

				else:
					raise NotImplementedError(
						'Not implemented latent walk type:' '{}'.format(stylegan_opts.latent))
		elif 'NN' in walk_type:
			self.walk = WalkNonLinearW(self.dim_z, self.step,
									   Nsliders, self.attrList).cuda()

		self.optimizers = torch.optim.Adam(self.walk.parameters(),
										   lr=self.lr,
										   betas=(0.5, 0.99))

		self.y = None
		self.z = None
		self.truncation = None
		self.walk_type = walk_type
		self.Nsliders = Nsliders

	def get_attr_idx(self):
		idxList = []
		for i in self.attrList:
			idxList.append(self.attrTable[i])
		return idxList

	def get_logits(self, inputs_dict, reshape=True):
		if self.stylegan_opts.latent == 'z':
			outputs_orig = self.module.netG(inputs_dict['z'])
		elif self.stylegan_opts.latent == 'w':
			# w in [18, N, 512]
			try:
				w = torch.stack(inputs_dict['w']).transpose(0, 1)
			except:
				w = inputs_dict['w'].transpose(0, 1)
			outputs_orig, _ = self.module.netG(w, input_is_latent=True)


		return outputs_orig

	def get_z_new(self, z, alpha):
		z_new = z
		for i in range(self.Nsliders):
			# TODO: PROBLEM HERE
			al = torch.unsqueeze(torch.Tensor(alpha[:, i]), axis=1)
			z_new = (z_new + al * self.w[:, :, i]).cuda()
		return z_new

	def get_z_new_tensor(self, z, alpha, name=None, trainEmbed=False, index_=None):
		z = z.squeeze()
		z_new = self.walk(z, name, alpha, index_)
		return z_new

	def get_w(self, z, is_single=False):
		w = self.module.netG.style(z)
		# W Space
		if is_single:
			return [w]
		# W+ Space
		return [w] * (self.step + 1) * 2

	def get_w_new_tensor(self, multi_ws, alpha, layers=None,
						 name=None, trainEmbed=False, index_=None):

		multi_ws_new = self.walk(multi_ws,
								 alpha=alpha,
								 layers=layers)
		return multi_ws_new

	def get_edit_loss(self, feed_dict):
		# L2 loss
		target = feed_dict['target']
		mask = feed_dict['mask_out']
		logit = feed_dict['logit']
		diff = (logit - target) * mask
		return torch.sum(diff.pow(2)) / torch.sum(mask)

	def get_reg_preds(self, logit):
		# Scene/Face
		preds = self.regressor(logit)[:, self.attrIdx]

		if len(preds.size()) == 1:
			preds = preds.unsqueeze(1)

		return preds

	def get_alphas(self, alpha_org, alpha_target):
		# [N, C]
		alpha_delta = alpha_target - alpha_org
		return alpha_delta

		# return alpha_target, alpha_delta #alpha_target, alpha_delta_new

	def get_bce_loss(self, pred, y, eps=1e-12):
		loss = -(y * pred.clamp(min=eps).log() + (1 - y) * (1 - pred).clamp(min=eps).log()).mean()
		return loss

	def get_reg_loss(self, feed_dict):
		logit = feed_dict['logit']
		alpha_gt = feed_dict['alpha'].to(torch.double)
		# Scene/Face
		preds = self.regressor(logit)[:, self.attrIdx]

		# BCE
		loss = self.get_bce_loss(preds, alpha_gt)
		return loss.mean()

	def get_content_loss(self, org_img, shifted_img):
		content_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']
		norm = Normalization().cuda()
		model = nn.Sequential(norm)

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
		self.optimizers.zero_grad()
		# D loss
		logit = feed_dict['logit']
		D_fake_result = self.module.netD(logit)		# B, 1
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

		if no_content_loss and no_gan_loss:
			# reg loss only
			loss = reg_loss
		else:
			# with 10 * reg loss
			loss = 10 * reg_loss
		if not no_content_loss:
			# with 0.05 * content loss
			loss += 0.05 * content_losses
		if not no_gan_loss:
			# with 0.05 * GAN loss
			loss += 0.05 * gan_loss

		loss.backward()
		self.optimizers.step()
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

	def load_multi_models_from_single(self, save_path_ws, save_path_gan,
						  trainEmbed=False, updateGAN=False,
						  single_transform_name=None,index=None):
		if updateGAN:
			print('Load GAN in %s' % save_path_gan)
			self.module = torch.load(save_path_gan)

		for i in range(len(save_path_ws)):
			walk_ckpt = torch.load(save_path_ws[i])
			self.walk.w[index[i]] = walk_ckpt.w[0]

	def get_reg_module(self):
		# Scene/Face, hard code resnet50 here
		model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False)
		model.fc = torch.nn.Linear(2048, 40)
		model = model.cuda()
		ckpt = torch.load(constants.reg_path)
		model.load_state_dict(ckpt['model'])
		"""
		If fine-tune or jointly train the classifier
		"""
		# optimizer.load_state_dict(ckpt['optm'])
		# return model, optimizer
		return model, None

	def get_vgg_module(self):
		vgg19 = models.vgg19(pretrained=True).features.cuda().eval()
		return vgg19

	def get_stylegan2_module(self):
		ckpt = torch.load(constants.g_path)
		from .networks import Generator
		gen = Generator(256, 512, 8)
		gen.load_state_dict(ckpt['g_ema'], strict=False)

		module = stylegan.StyleGAN(lr=self.lr)
		module.netG = gen.cuda().eval()
		print('Finish loading the pretrained model')
		return module

	def clip_ims(self, ims):
		return np.uint8(np.clip(((ims + 1) / 2.0) * 255, 0, 255))

	def apply_alpha(self, graph_inputs, alpha_to_graph,
						layers=None, name=None,
						trainEmbed=False, index_=None,
						given_w=None):
		with torch.no_grad():
			zs_batch = graph_inputs['z']  # tensor.cuda() # [Batch, DIM_Z]
			if self.stylegan_opts.latent == 'w':
				if given_w!=None:
					latent_w = given_w
				else:
					latent_w = self.get_w(zs_batch)


				feed_dict = {'w': latent_w}
				out_zs = self.get_logits(feed_dict)

				alpha_org = self.get_reg_preds(out_zs)					# [B, C]
				# alpha_delta = torch.zeros_like(alpha_org).cuda()		# alpha_org - alpha_org
				alpha_delta = self.get_alphas(alpha_org, torch.Tensor(alpha_to_graph).cuda())
				if index_!=None:
					# alpha_to_graph [B, 1]
					# alpha_delta [B, attrList]
					# for j in range(len(self.attrIdx)):
					if len(self.attrIdx) == len(self.attrTable):
						# alpha_delta[:, index_] = torch.Tensor(alpha_to_graph[:, 0]).cuda() - alpha_org[:, index_]
						alpha_delta[:, index_] = torch.Tensor(alpha_to_graph).cuda() - alpha_org[:, index_]
					else:
						i = self.attrIdx.index(index_)
						alpha_delta[:, i] = torch.Tensor(alpha_to_graph[:, 0]).cuda() - alpha_org[:, i]
						# alpha_delta[:, i] = torch.Tensor(alpha_to_graph).cuda() - alpha_org[:, i]

			if self.stylegan_opts.latent == 'z':
				z_new = self.get_z_new_tensor(zs_batch, alpha_to_graph, name, trainEmbed=trainEmbed, index_=index_)
				best_inputs = {'z': z_new}
				best_im_out = self.get_logits(best_inputs)

			elif self.stylegan_opts.latent == 'w':
				latent_w_new = self.get_w_new_tensor(latent_w,
													 alpha_delta,
													 layers=layers,
													 name=name,
													 trainEmbed=trainEmbed,
													 index_=index_)

				best_inputs = {'w': latent_w_new}
				best_im_out = self.get_logits(best_inputs)

			else:
				raise ('Non implemented')
		return best_im_out, alpha_org, out_zs


	def vis_multi_image_batch_alphas(self, graph_inputs, filename,
									 alphas_to_graph, alphas_to_target,
									 batch_start,
									 layers=None,
									 name=None, wgt=False, wmask=False,
									 trainEmbed=False, computeL2=False,
									 given_w=None, index_=None):

		zs_batch = graph_inputs['z']  # numpy
		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []

		for ag, at in zip(alphas_to_graph, alphas_to_target):
			input_test = {'z': torch.Tensor(zs_batch).cuda()}
			best_im_out, alpha_org, out_zs = self.apply_alpha(input_test, ag, name=name, layers=layers,
											trainEmbed=trainEmbed, given_w=given_w, index_=index_)

			best_im_out = best_im_out.detach().cpu().numpy()
			best_im_out = np.uint8(np.clip(((best_im_out + 1) / 2.0) * 255, 0, 255))
			ims_transformed.append(best_im_out)

		img_org = out_zs.detach().cpu().numpy()

		for ii in range(zs_batch.shape[0]):

			if index_!=None and len(self.attrList)>1:
				a = alpha_org[ii, index_].item()
			else:
				a = alpha_org[ii].item()
			if wmask:
				arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
										  in zip(ims_transformed, ims_mask)], axis=0)
			else:
				ims_transformed_new = ims_transformed
				arr_transform = np.stack([x[ii, :, :, :] for x in ims_transformed_new], axis=0)
				# arr_transform = np.stack([x[ii, :, :, :] for x in
				# 						  ims_transformed], axis=0)
			# arr_gt = self.clip_ims(arr_gt)
			# arr_transform = self.clip_ims(arr_transform)
			ims = arr_transform
			filename = filename_base + '_sample{}'.format(ii + batch_start)
			if wgt:
				filename += '_wgt'
			if wmask:
				filename += '_wmask'
			if ims.shape[1] == 1 or ims.shape[1] == 3:
				# N C W H -> N W H C
				ims = np.transpose(ims, [0, 2, 3, 1])
			# ims = np.squeeze(ims)
			filename = filename +'_%.2f' % a
			print('Save in ', filename)
			image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

			"""
			If save original image
			"""
			# temp = filename.split('/')
			# path = '/'.join(temp[:-1]) +  '/org_img/' +  temp[-1]
			# path_dir = '/'.join(temp[:-1]) +  '/org_img'
			# if not os.path.exists(path_dir):
			# 	print('pt: ', path_dir)
			# 	os.mkdir(path_dir)
			#
			# ut.save_image( out_zs[ii], path+'_org.jpg', nrow=1, normalize=True,
			# 	range=(-1, 1), padding=0)


	def vis_multi_image_batch_alphas_compute_multi_attr(self, graph_inputs, filename,
									 alphas_to_graph, alphas_to_target,
									 batch_start,
									 layers=None,
									 name=None, wgt=False, wmask=False,
									 trainEmbed=False, computeL2=False,
									 given_w=None, index_=None):

		zs_batch = graph_inputs['z']  # numpy
		ims_transformed = []
		filename_base = filename
		multi_attr = [[],[],[]]
		attri_org = [[],[],[]]
		imgs = [[],[],[]]
		orgs =[[],[],[]]
		with torch.no_grad():
			for ag1, at1 in zip(alphas_to_graph, alphas_to_target):

					ims_transformed = []
					input_test = {'z': torch.Tensor(zs_batch).cuda()}
					# for ag2, at2 in zip(alphas_to_graph, alphas_to_target):

					# best_im_out, alpha_org, out_zs = self.apply_alpha_combine(input_test, [ag1, ag2],
					# 								name=name, layers=layers,
					# 								trainEmbed=trainEmbed, given_w=given_w, index_=index_)

					best_im_out, alpha_org, out_zs = self.apply_alpha(input_test, ag1,
																			  name=name, layers=layers,
																			  trainEmbed=trainEmbed,
																			  given_w=given_w, index_=index_)

					pred_attr = self.regressor(best_im_out).detach().cpu().numpy() # N, 40
					org = self.regressor(out_zs).detach().cpu().numpy()

					best_im_out = best_im_out.detach().cpu().numpy()
					best_im_out = np.uint8(np.clip(((best_im_out + 1) / 2.0) * 255, 0, 255))
					ims_transformed.append(best_im_out)
					out_zs_org = out_zs.cpu()
					out_zs = np.uint8(np.clip(((out_zs.cpu().numpy() + 1) / 2.0) * 255, 0, 255))
					if type(index_) == int:
						index_list = [index_]
					else:
						index_list = index_
					for i in range(pred_attr.shape[0]):
						# print('check attri: ', pred_attr[i, index_])
						if np.abs(pred_attr[i, index_list[0]] - org[i, index_list[0]]) <= 0.3:
							multi_attr[0].append(pred_attr[i])
							attri_org[0].append(org[i])
							imgs[0].append(best_im_out[i])
							orgs[0].append(out_zs[i])

						elif np.abs(pred_attr[i, index_list[0]] - org[i, index_list[0]]) <= 0.6:
							multi_attr[1].append(pred_attr[i])
							attri_org[1].append(org[i])
							imgs[1].append(best_im_out[i])
							orgs[1].append(out_zs[i])

						elif np.abs(pred_attr[i, index_list[0]] - org[i, index_list[0]]) <= 1:
							multi_attr[2].append(pred_attr[i])
							attri_org[2].append(org[i])
							imgs[2].append(best_im_out[i])
							orgs[2].append(out_zs[i])

					# for ii in range(zs_batch.shape[0]):
					# 	# print('a :', a)
					# 	arr_transform = np.stack([x[ii, :, :, :] for x in ims_transformed], axis=0)
					# 	ims = arr_transform
					# 	filename = filename_base + '_idx{}_idx{}_{}_sample{}'.format(index_[0],
					# 														 index_[0],
					# 														 at1,
					# 														 ii + batch_start)
					# 	# print('at1 at2', at1, at2, filename)
					# 	if ims.shape[1] == 1 or ims.shape[1] == 3:
					# 		ims = np.transpose(ims, [0, 2, 3, 1])
					# 	image.save_im(image.imgrid(ims, cols=1), filename)
					#
					# 	temp = filename.split('/')
					# 	import os
					# 	path = '/'.join(temp[:-1]) + '/org_img/' + temp[-1]
					# 	path_dir = '/'.join(temp[:-1]) + '/org_img'
					# 	if not os.path.exists(path_dir):
					# 		print('pt: ', path_dir)
					# 		os.mkdir(path_dir)
					# 	ut.save_image(
					# 		out_zs_org[ii],
					# 		path+'_org.jpg',
					# 		nrow=1,
					# 		normalize=True,
					# 		range=(-1, 1),
					# 		padding=0)


			return multi_attr, attri_org, imgs, orgs

	def apply_alpha_combine(self, graph_inputs, alpha_to_graph,
						layers=None, name=None,
						trainEmbed=False, index_=None,
						given_w=None):
		with torch.no_grad():
			zs_batch = graph_inputs['z']  # tensor.cuda() # [Batch, DIM_Z]
			if self.stylegan_opts.latent == 'w':
				if given_w == None:
					latent_w = self.get_w(zs_batch)
				else:
					latent_w = given_w

				feed_dict = {'w': latent_w}
				out_zs = self.get_logits(feed_dict)
				alpha_org = self.get_reg_preds(out_zs)					# [B, C]
				alpha_delta = torch.zeros_like(alpha_org).cuda()		# alpha_given - alpha_org
				ct = 0

				for k, i in enumerate(index_):
					# print('Check alpha for index:', alpha_to_graph[ct][:, 0], i, ct)
					alpha_delta[:, i] = torch.Tensor(alpha_to_graph[ct][:, 0]).cuda() - alpha_org[:, i]
					ct += 1

			if self.stylegan_opts.latent == 'z':
				z_new = self.get_z_new_tensor(zs_batch, alpha_to_graph, name, trainEmbed=trainEmbed, index_=index_)
				best_inputs = {'z': z_new}
				print('zs z new: ', zs_batch[0, :4], z_new[0, :4])
				best_im_out = self.get_logits(best_inputs)

			elif self.stylegan_opts.latent == 'w':
				latent_w_new = self.get_w_new_tensor(latent_w,
													 alpha_delta,
													 layers=layers,
													 name=name,
													 trainEmbed=trainEmbed,
													 index_=index_)

				best_inputs = {'w': latent_w_new}
				best_im_out = self.get_logits(best_inputs)

			else:
				raise ('Non implemented')
			return best_im_out, alpha_org, out_zs


	def vis_multi_image_batch_alphas_combine(self, graph_inputs, filename,
											 alphas_to_graph, alphas_to_target,
											 batch_start,
											 layers=None,
											 name=None, wgt=False, wmask=False,
											 trainEmbed=False, computeL2=False,
											 given_w=None, index_=None):

		zs_batch = graph_inputs['z']  # numpy
		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []
		# imgs_new = [[]*len(zs_batch.shape[0])]
		with torch.no_grad():

			for ag1, at1 in zip(alphas_to_graph, alphas_to_target):
				ims_transformed = []
				input_test = {'z': torch.Tensor(zs_batch).cuda()}
				for ag2, at2 in zip(alphas_to_graph, alphas_to_target):
					best_im_out, alpha_org, out_zs = self.apply_alpha_combine(input_test, [ag1, ag2],
																		name=name, layers=layers,
																		trainEmbed=trainEmbed,
																		given_w=given_w, index_=index_)
					best_im_out = best_im_out.detach().cpu().numpy()
					best_im_out = np.uint8(np.clip(((best_im_out + 1) / 2.0) * 255, 0, 255))
					ims_transformed.append(best_im_out)

					# imgs_new[i].append(best_im_out)

				for ii in range(zs_batch.shape[0]):
					arr_transform = np.stack([x[ii, :, :, :] for x in ims_transformed], axis=0)
					ims = arr_transform
					filename = filename_base + '_idx{}_idx{}_{}_sample{}'.format(index_[0],
																		 index_[1],
																		 at1,
																		 ii + batch_start)
					# print('at1 at2', at1, at2, filename)
					if ims.shape[1] == 1 or ims.shape[1] == 3:
						ims = np.transpose(ims, [0, 2, 3, 1])
					image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

					temp = filename.split('/')
					import os
					path = '/'.join(temp[:-1]) + '/org_img/' + temp[-1]
					path_dir = '/'.join(temp[:-1]) + '/org_img'
					if not os.path.exists(path_dir):
						print('pt: ', path_dir)
						os.mkdir(path_dir)
					ut.save_image(
						out_zs[ii],
						path+'_org.jpg',
						nrow=1,
						normalize=True,
						range=(-1, 1),
						padding=0)

	def vis_image_batch(self, graph_inputs, filename,
						batch_start, wgt=False, wmask=False, num_panels=7):
		raise NotImplementedError('Subclass should implement vis_image_batch')


class BboxTransform(TransformGraph):
	def __init__(self, *args, **kwargs):
		TransformGraph.__init__(self, *args, **kwargs)

	def get_distribution_statistic(self, img, channel=None):
		raise NotImplementedError('Subclass should implement get_distribution_statistic')

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
		"""
		Not being used in the end
		"""

		random_seed = 2
		rnd = np.random.RandomState(random_seed)
		inputs = graph_input(self, num_samples, seed=random_seed)
		batch_size = constants.BATCH_SIZE
		model_samples = []
		for a in self.test_alphas():
			distribution = []
			start = time.time()
			print("Computing attribute statistic for alpha={:0.2f}".format(a))
			for batch_num, batch_start in enumerate(range(0, num_samples, batch_size)):
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
