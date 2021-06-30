from __future__ import absolute_import
import torch, torchvision
import numpy as np
from . import constants, pggan_128
from .model import Generator
from . import model
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
			'zoom': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).cuda()),
			'shiftx': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).cuda()),
			'rotate2d': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).cuda())})

	def forward(self, z, name, alpha, index_):
		z_new = z  #.cpu()
		for i in range(self.Nsliders):
			z_new = z_new + self.w_embed[name][index_, :, :, i]
			# al = torch.unsqueeze(alpha[:, i], axis=1)
			# z_new = z_new + al.cpu() * self.w_embed[name][index_, :, :, i]
		return z_new

class walk_linear(nn.Module):
	def __init__(self, dim_z, Nsliders):
		super(walk_linear, self).__init__()
		self.dim_z = dim_z
		self.Nsliders = Nsliders

		self.w_embed = nn.ParameterDict({
			'zoom': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders]))),
			'shiftx': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders]))),
			'rotate2d': nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])))})

	def forward(self, z, name, alpha, index_):
		z_new = z.cpu()
		for i in range(self.Nsliders):
			al = torch.unsqueeze(alpha[:, i], axis=1)
			z_new = z_new + al.cpu() * self.w_embed[name][:, :, i]
		return z_new.cuda()


class TransformGraph():
	def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f,
				 trainEmbed):
		assert (loss_type in ['l2', 'lpips']), 'unimplemented loss'

		# module inputs
		self.lr = lr
		self.useGPU = constants.useGPU
		self.module = self.get_pgan_module()

		self.dim_z = constants.DIM_Z
		self.Nsliders = Nsliders = nsliders
		self.img_size = constants.resolution
		self.num_channels = constants.NUM_CHANNELS
		# self.CRITIC_ITERS = CRITIC_ITERS = constants.CRITIC_ITERS
		# self.OUTPUT_DIM = constants.OUTPUT_DIM
		self.BATCH_SIZE = constants.BATCH_SIZE
		self.LAMBDA = 0.1
		self.trainEmbed = trainEmbed

		self.BCE_loss = nn.BCELoss()
		self.BCE_loss_logits = nn.BCEWithLogitsLoss()
		print('LR for w', self.lr)
		# walk pattern
		if walk_type == 'linear':
			if self.trainEmbed == True:
				self.walk = walk_embed(self.dim_z, Nsliders)
			else:
				print('Walk in linear')
				self.walk = walk_linear(self.dim_z, Nsliders)

			self.optimizers_embed = {}

			self.optimizers_embed['zoom'] = torch.optim.Adam([self.walk.w_embed['zoom']], lr=10 * self.lr,
																  betas=(0.5, 0.99))
			self.optimizers_embed['shiftx'] = torch.optim.Adam([self.walk.w_embed['shiftx']], lr=10 * self.lr,
																 betas=(0.5, 0.99))
			self.optimizers_embed['rotate2d'] = torch.optim.Adam([self.walk.w_embed['rotate2d']], lr=10 * self.lr,
																 betas=(0.5, 0.99))

			"""
			self.w_embed = {}
			self.w_embed['zoom'] = torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).requires_grad_()
			self.w_embed['shiftx'] = torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).requires_grad_()
			self.w_embed['shifty'] = torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).requires_grad_()
			self.w_embed['rotate2d'] = torch.Tensor(np.random.normal(0.0, 0.02, [15, 1, self.dim_z, Nsliders])).requires_grad_()

			self.optimizers_embed = {}
			self.optimizers_embed['zoom'] = torch.optim.Adam([self.w_embed['zoom']], lr=10 * self.lr, betas=(0.5, 0.99))
			self.optimizers_embed['shiftx'] = torch.optim.Adam([self.w_embed['shiftx']], lr=10 * self.lr, betas=(0.5, 0.99))
			self.optimizers_embed['shifty'] = torch.optim.Adam([self.w_embed['shifty']], lr=10 * self.lr, betas=(0.5, 0.99))
			self.optimizers_embed['rotate2d'] = torch.optim.Adam([self.w_embed['rotate2d']], lr=10 * self.lr, betas=(0.5, 0.99))

			
			# WS FOR MULTI TRANSFORMATIONS
			# self.ws = {}
			# self.ws['zoom'] = torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])).cuda().requires_grad_()
			# self.ws['shiftx'] = torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])).cuda().requires_grad_()
			# self.ws['shifty'] = torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])).cuda().requires_grad_()
			# self.ws['rotate2d'] = torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])).cuda().requires_grad_()
			# 
			# self.optimizers = {}
			# self.optimizers['zoom'] = torch.optim.Adam([self.ws['zoom']], lr=10*self.lr, betas=(0.5, 0.99))
			# self.optimizers['shiftx'] = torch.optim.Adam([self.ws['shiftx']], lr=10 * self.lr, betas=(0.5, 0.99))
			# self.optimizers['shifty'] = torch.optim.Adam([self.ws['shifty']], lr=10 * self.lr, betas=(0.5, 0.99))
			# self.optimizers['rotate2d'] = torch.optim.Adam([self.ws['rotate2d']], lr=10 * self.lr, betas=(0.5, 0.99))
			"""
		else:
			raise NotImplementedError('Not implemented walk type:' '{}'.format(walk_type))

		# TODO: lr_w
		# self.optimizer = train_optimizer = torch.optim.Adam([self.w], lr= 10*self.lr, betas=(0.5, 0.99))

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

	#TODO:
	def get_logits(self, inputs_dict, reshape=False):
		outputs_orig = self.module.netG(inputs_dict['z'])
		return outputs_orig


	def get_z_new(self, z, alpha):
		if self.walk_type == 'linear' or self.walk_type == 'NNz':
			for i in range(self.Nsliders):
				# TODO: PROBLEM HERE
				al = torch.unsqueeze(torch.Tensor(alpha[:, i]), axis=1)
				z_new = (z + al * self.w[:, :, i]).cuda()
		return z_new

	def get_z_new_tensor(self, z, alpha, name=None, trainEmbed=False, index_=None):
		z = z.squeeze()
		z_new = self.walk(z, name, alpha, index_)
		# return z_new.cuda()
		return z_new

	# def get_z_new_tensor(self, z, alpha, name=None, trainEmbed=False, index_=None):
	# 	z = z.squeeze()
	# 	if self.walk_type == 'linear' or self.walk_type == 'NNz':
	# 		for i in range(self.Nsliders):
	# 			al = torch.unsqueeze(alpha[:, i], axis=1)
	# 			if name:
	# 				if trainEmbed:
	# 					# print('get z new with Embed!', name)
	# 					z_new = z.cpu() + al.cpu() * self.w_embed[name][index_, :, :, i].cpu()
	# 				else:
	# 					print('get z new!', name)
	# 					z_new = z.cpu() + al.cpu() * self.ws[name][ :, :, i].cpu()
	# 			else:
	# 				print('Single W')
	# 				z_new = z.cpu() + al.cpu() * self.w[:, :, i].cpu()
	# 	return z_new.cuda()

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
		return torch.sum(diff.pow(2))/torch.sum(mask)

	def update_params(self, loss):
		print('Before w: ', self.w[0,:3])
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		print('Update w: ', self.w[0, :3])

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

			# D real
			D_real_result = self.module.netD(x_real).squeeze()
			D_real_result = D_real_result.mean() - 0.001 * (D_real_result ** 2).mean()
			D_real_result.backward(self.module.mone, retain_graph=True)

			# D fake
			D_fake_result = self.module.netD(logit).squeeze()
			D_fake_result = D_fake_result.mean()
			D_fake_result.backward(self.module.one, retain_graph=True)

			# TRAIN WITH GRADIENT PENALTY
			# gp = gradient_penalty(functools.partial(self.module.netD), x_real, logit,
			# 					  gp_mode='1-gp',
			# 					  sample_mode='line')
			# gradient_penalty = calc_gradient_penalty(self.module.netD, x_real.data, logit.data, self.BATCH_SIZE)

			eps = torch.rand(constants.BATCH_SIZE, 1, 1, 1).cuda()
			x_hat = eps * x_real.data + (1 - eps) * logit.data
			x_hat = Variable(x_hat, requires_grad=True)
			hat_predict = self.module.netD(x_hat)
			grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
			grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
			grad_penalty = 10 * grad_penalty
			grad_penalty.backward(retain_graph=True)

			# grad_loss_val = grad_penalty.data
			# disc_loss_val = (real_predict - fake_predict).data
			self.module.optimizerD.step()

			# Update G
			self.module.optimizerG.zero_grad()
			new_logit = self.get_logits(feed_dict)
			feed_dict['logit'] = new_logit

			D_fake_result = self.module.netD(new_logit).squeeze()
			G_train_loss = self.BCE_loss_logits(D_fake_result, y_real)

			Edit_loss = self.get_edit_loss(feed_dict)
			G_train_loss += self.LAMBDA * Edit_loss
			G_train_loss.backward(retain_graph=True)
			self.module.optimizerG.step()
			self.module.accumulate(self.module.g_running, self.module.netG)

		# Update w
		# if trainEmbed:
		print('Update W')

		for name in self.optimizers_embed.keys():
			self.optimizers_embed[name].zero_grad()
			new_logit = self.get_logits(feed_dict)
			feed_dict['logit'] = new_logit
			Edit_loss = self.get_edit_loss(feed_dict)
			# print('Edit_loss2: ', Edit_loss)

			Edit_loss.backward(retain_graph=True)
			self.optimizers_embed[name].step()

		return Edit_loss


	def save_model(self, save_path_w, save_path_gan):
			print('Save W and GAN in %s and %s' % (save_path_w, save_path_gan))
			torch.save(self.module, save_path_gan)
			# self.module.save(save_path_gan)
			np.save(save_path_w, self.w.detach().cpu().numpy())

	def save_multi_models(self, save_path_w, save_path_gan, trainEmbed=False, updateGAN=True, single_transform_name=None):
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
					np.save(cur_path_w,self.walk.w_embed[single_transform_name].detach().cpu().numpy())
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

	# def get_pgan_module(self):
	# 	print('Loading PGGAN module')
	# 	# 256
	# 	module = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
	# 							pretrained=True,
	# 							model_name='celebAHQ-256',
	# 							useGPU=self.useGPU)
	# 	return module


	def get_pgan_module(self):
		print('Loading PGGAN module')
		# generator = model.Generator(511, 1).cuda().train(False)
		module = pggan_128.PGGAN(lr=self.lr)
		# load_pretrain
		base_dir = '/home/peiye/ImageEditing/vision_16_pgan/checkpoint/'
		ckpt = torch.load(base_dir + '600000_dict.model')

		#base_dir = '/home/peiye/ImageEditing/progressive-gan-pytorch/checkpoint/'
		print('Start loading PGGANA_celebA module in %s' % (base_dir))
		# g_pretrain = base_dir + 'generator_param_final_epoch100.pkl'
		# d_pretrain = base_dir + 'discriminator_param_final_epoch100.pkl'
		#ckpt = torch.load(base_dir + '510000_dict.model')

		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in ckpt['G'].items():
			name = k[7:]
			new_state_dict[name] = v
		module.netG.load_state_dict(new_state_dict)

		# ckpt_g = torch.load(base_dir + '520000.model').state_dict()
		# new_state_dict = OrderedDict()
		# for k, v in ckpt_g.items():
		# 	name = k[7:]
		# 	new_state_dict[name] = v
		# generator.load_state_dict(new_state_dict)

		new_state_dict = OrderedDict()
		for k, v in ckpt['D'].items():
			name = k[7:]
			new_state_dict[name] = v
		module.netD.load_state_dict(new_state_dict)

		# module.netG.load_state_dict(ckpt['G'], strict=False)
		# module.netD.load_state_dict(ckpt['D'], strict=False)
		print('Finish loading the pretrained model')
		return module


	def clip_ims(self, ims):
		return np.uint8(np.clip(((ims + 1) / 2.0) * 256, 0, 255))


	def apply_alpha(self, graph_inputs, alpha_to_graph, name=None, trainEmbed=False, index_=None):
		zs_batch = graph_inputs['z'] #tensor.cuda() # [Batch, DIM_Z]
		# print(alpha_to_graph)
		if self.walk_type == 'linear':
			# best_inputs = {self.z: zs_batch, self.alpha: alpha_to_graph}
			# best_im_out = self.sess.run(self.transformed_output, best_inputs)

			alpha_to_graph = torch.tensor(alpha_to_graph).float().cuda()
			z_new = self.get_z_new_tensor(zs_batch, alpha_to_graph, name,  trainEmbed=trainEmbed, index_=index_)
			best_inputs = {'z': z_new}
			best_im_out = self.get_logits(best_inputs)
			return best_im_out

		# TODO:
		elif self.walk_type.startswith('NN'):
			# alpha_to_graph is number of steps and direction
			direction = np.sign(alpha_to_graph)
			num_steps = np.abs(alpha_to_graph)
			# embed()
			single_step_alpha = self.N_f + direction
			# within the graph range, we can compute it directly
			if 0 <= alpha_to_graph + self.N_f <= self.N_f * 2:
				latent_space_out = self.sess.run(self.latent_space_new, {
					self.z: zs_batch, self.alpha:
						alpha_to_graph + self.N_f})
			# # sanity check
			# zs_next = zs_batch
			# for n in range(num_steps):
			#     feed_dict = {self.z: zs_next, self.alpha: single_step_alpha}
			#     zs_next = self.sess.run(self.z_new, feed_dict=feed_dict)
			# zs_test = zs_next
			# assert(np.allclose(zs_test, zs_out))
			else:
				# print("recursive zs for {} steps".format(alpha_to_graph))
				latent_space_batch = self.sess.run(self.latent_space, {self.z: zs_batch})
				latent_space_next = latent_space_batch
				for n in range(num_steps):
					feed_dict = {self.latent_space: latent_space_next, self.alpha: single_step_alpha}
					latent_space_next = self.sess.run(self.latent_space_new, feed_dict=feed_dict)
				latent_space_out = latent_space_next
			# already taken n steps at this poin
			best_inputs = {self.latent_space_new: latent_space_out}  # , self.alpha: self.N_f}
			best_im_out = self.sess.run(self.transformed_output,
										best_inputs)
			return best_im_out

	def L2_loss(self, img1, img2, mask):
		return np.sum(np.square((img1 - img2)*mask), (1, 2, 3))

	def vis_image_batch_alphas(self, graph_inputs, filename,
							   alphas_to_graph, alphas_to_target,
							   batch_start, name=None, wgt=False, wmask=False,
							   trainEmbed=False, computeL2=True):

		zs_batch = graph_inputs['z'] #numpy
		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []

		index_ = 0
		L2_loss = {}
		for ag, at in zip(alphas_to_graph, alphas_to_target):

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
									 get_target_np,
								   alphas_to_graph, alphas_to_target,
								   batch_start, name=None, wgt=False, wmask=False,
									trainEmbed=False,computeL2=False):
		# TODO:
		# CHANGE!!
		zs_batch = graph_inputs['z'] #numpy

		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []
		L2_loss = {}
		index_ = 0
		for ag, at in zip(alphas_to_graph, alphas_to_target):

			input_test = {'z': torch.Tensor(zs_batch).cuda()}
			out_input_test = self.get_logits(input_test)
			output_zs_256 = F.interpolate(out_input_test, size=256)  # T

			out_input_test2 = output_zs_256.detach().cpu().numpy()  # on Cuda
			target_fn, mask_out = get_target_np(out_input_test2, at)
			target_fn = np.uint8(np.clip(((target_fn + 1) / 2.0) * 255, 0, 255))

			best_im_out = self.apply_alpha(input_test, ag, name, trainEmbed=trainEmbed, index_=index_)
			best_im_out = F.interpolate(best_im_out, size=256).detach().cpu().numpy()
			best_im_out = np.uint8(np.clip(((best_im_out + 1) / 2.0) * 255, 0, 255))

			ims_target.append(target_fn)
			ims_transformed.append(best_im_out)
			ims_mask.append(mask_out)

			if computeL2:
				L2_loss[at] = self.L2_loss(target_fn, best_im_out, mask_out)
			index_ += 1

		if computeL2:
			return L2_loss

		print('wgt: ', wgt)

		for ii in range(zs_batch.shape[0]):
			arr_gt = np.stack([x[ii, :, :, :] for x in ims_target], axis=0)
			if wmask:
				arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
										  in zip(ims_transformed, ims_mask)], axis=0)
			else:
				arr_transform = np.stack([x[ii, :, :, :] for x in
										  ims_transformed], axis=0)
			# arr_gt = self.clip_ims(arr_gt)
			# arr_transform = self.clip_ims(arr_transform)

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
			if ims.shape[1] == 1 or ims.shape[1] == 3:
				# N C W H -> N W H C
				ims = np.transpose(ims, [0, 2, 3, 1])
				# ims = np.squeeze(ims)
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
            print("Sampled {} images in {:0.2f} min".format(num_samples, (end-start)/60))
            model_samples.append(distribution)

        model_samples = np.array(model_samples)
        return model_samples


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
'''
