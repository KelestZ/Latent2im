import logging
import utils.logging
from utils import util, image
import sys
import torch
import graphs
import time
import importlib
import os
import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
# from transform_graph import get_transform_graphs
# from IPython import embed
import torchvision.datasets as dset

"""
Usage: 
----
Multi attr
###
python train_multi_attr.py --model stylegan_v2_real --transform scene \
        --num_samples 20000 --learning_rate 1e-4 --latent w --attrList night,dark \
        --walk_type linear --loss l2 --gpu 0 --overwrite_config --prefix 'DarkNight' \
        --models_dir ./models_scene_256_multi_attr

###
### Note: you may assign different attributes for the ``attrList" flag

python train_multi_attr.py --model stylegan_v2_real --transform face \
        --num_samples 20000 --learning_rate 1e-4 --latent w --attrList Smiling \
        --walk_type linear --loss l2 --gpu 0 \
        --models_dir ./models_celeba_256_multi_attr


"""


def train(model_name, graphs, graph_inputs, output_dir, attrList,
		  layers=None,
		  save_freq=100, trainEmbed=False, updateGAN=False, single_transform_name=None,
		  opt=None):
	# configure logging file
	logging_file = os.path.join(output_dir, 'log.txt')
	utils.logging.configure(logging_file, append=False)

	Loss_sum = 0
	n_epoch = 3

	optim_iter = 0
	batch_size = constants.BATCH_SIZE
	num_samples = graph_inputs['z'].shape[0]
	loss_values = []

	for epoch in range(n_epoch):
		if updateGAN:
			it = iter(train_loader)
			ITERS = min(len(train_loader), num_samples // batch_size)
		else:
			ITERS = (num_samples // batch_size)

		graph_inputs = graph_util.graph_input(graphs, num_samples, seed=epoch)
		print('Number of the training epochs and iterations: ', n_epoch, ITERS)

		for i in range(ITERS):
			batch_start = i * batch_size
			start_time = time.time()
			s = slice(batch_start, min(num_samples, batch_start + batch_size))
			graph_inputs_batch = util.batch_input(graph_inputs, s)

			zs_batch = graph_inputs_batch['z']

			graph_inputs_batch_cuda = {}
			graph_inputs_batch_cuda['z'] = torch.Tensor(graph_inputs_batch['z']).cuda()

			z_global = graph_inputs_batch_cuda['z']
			w_global = graphs.get_w(z_global)
			graph_inputs_batch_cuda['w'] = w_global

			out_zs = graphs.get_logits(graph_inputs_batch_cuda)
			alpha_org = graphs.get_reg_preds(out_zs)		# N, C

			# alphas_g = {}
			# alphas_t = {}
			alphas_reg = []

			if updateGAN:
				current_batch, current_label = next(it)
				target_global = out_zs
				mask_global = None
				edit_real_global = current_batch.numpy()
			##
			# [-1, 1]
			alpha_for_graph, alpha_for_target, index_ = graphs.get_train_alpha(zs_batch, N_attr=len(attrList), trainEmbed=trainEmbed)
			alphas_reg.append(alpha_for_graph)

			if not isinstance(alpha_for_graph, list):
				alpha_for_graph = [alpha_for_graph]
				alpha_for_target = [alpha_for_target]

			# alphas_g[name] = alpha_for_graph
			# alphas_t[name] = alpha_for_target

			for ag, at in zip(alpha_for_graph, alpha_for_target):

				ag = torch.tensor(ag).float().cuda()
				alpha_target, alpha_delta_new = graphs.get_alphas(alpha_org, ag)
				ag = alpha_delta_new
				w_new = graphs.get_w_new_tensor(w_global, ag,
												layers=layers)

				transformed_inputs = graph_inputs_batch_cuda
				transformed_inputs['w'] = w_new
				# transformed_inputs['z'] = z_new
				transformed_output = graphs.get_logits(transformed_inputs)
				w_global = w_new

				feed_dict = {}

				# feed_dict['at'] = at
				# feed_dict['z'] = z_new
				feed_dict['w'] = w_global
				feed_dict['org'] = out_zs
				feed_dict['logit'] = transformed_output

			####
			feed_dict['alpha'] = alpha_target

			curr_loss = graphs.optimizeParametersAll(feed_dict,
													 trainEmbed=trainEmbed,
													 updateGAN=updateGAN,
													 no_content_loss=opt.no_content_loss,
													 no_gan_loss=opt.no_gan_loss
													 )
			curr_loss_item = curr_loss.detach().cpu().item()
			Loss_sum = Loss_sum + curr_loss_item
			loss_values.append(curr_loss_item)
			elapsed_time = time.time() - start_time

			logging.info('T, epc, bst, lss, alpha night: {}, {}, {}, {}, {}'.format(
				elapsed_time, epoch, batch_start, curr_loss, round(at[0], 2)))

			if (optim_iter % save_freq == 0):
				make_samples(out_zs, output_dir, epoch, optim_iter * batch_size, batch_size,
							 name='org_%.2f' % (round(at[0], 2)))
				make_samples(transformed_output, output_dir, epoch, optim_iter * batch_size, batch_size,
							 name='logit_%.2f'%(round(at[0], 2)))
			optim_iter = optim_iter + 1

		graphs.save_multi_models('{}/model_w_{}'.format(output_dir, epoch),
								 '{}/model_gan_{}.ckpt'.format(output_dir, epoch),
								 trainEmbed=trainEmbed,
								 updateGAN=updateGAN)

		# make_samples(transformed_output, output_dir, epoch, optim_iter * batch_size, batch_size, name='logit')

	graphs.save_multi_models('{}/model_w_{}_final'.format(output_dir, n_epoch),
							 '{}/model_gan_{}_final.ckpt'.format(output_dir, n_epoch),
							 trainEmbed=trainEmbed,
							 updateGAN=updateGAN)

	return loss_values

def make_samples(img_tensor, output_dir, epoch, optim_iter, batch_size, pre_path='results', name='test'):
	try:
		if img_tensor.is_cuda:
			img_tensor = img_tensor.detach().cpu().numpy()
			img_tensor = np.uint8(np.clip(((img_tensor + 1) / 2.0) * 255, 0, 255))
			if img_tensor.shape[1] == 1 or img_tensor.shape[1] == 3:
				img_tensor = np.transpose(img_tensor, [0, 2, 3, 1])
			image.save_im(image.imgrid(img_tensor, cols=4),
						  '{}/{}/{}_{}_{}'.format(output_dir, pre_path, epoch, optim_iter, name))
	except:
		img_tensor = np.uint8(np.clip(((img_tensor + 1) / 2.0) * 255, 0, 255))
		if img_tensor.shape[1] == 1 or img_tensor.shape[1] == 3:
			img_tensor = np.transpose(img_tensor, [0, 2, 3, 1])
		image.save_im(image.imgrid(img_tensor, cols=4),
					'{}/{}/{}_{}_{}'.format(output_dir, pre_path, epoch, optim_iter, name))


if __name__ == '__main__':

	opt = TrainOptions().parse()
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

	output_dir = opt.output_dir
	if not os.path.exists(os.path.join(output_dir, 'results')):
		os.makedirs(os.path.join(output_dir, 'results'))
	graph_kwargs = util.set_graph_kwargs(opt)
	print('graph_kwargs1: ', opt.model, opt.transform, graph_kwargs.keys())

	graph_util = importlib.import_module('graphs.' + opt.model + '.graph_util')
	constants = importlib.import_module('graphs.' + opt.model + '.constants')
	model = graphs.find_model_using_name(opt.model, opt.transform)

	g = model(**graph_kwargs)

	num_samples = opt.num_samples
	graph_inputs = graph_util.graph_input(g, num_samples, seed=0)

	if opt.suffix:
		name = opt.suffix
	else:
		name = None

	# attrList = opt.attrList.split(',')
	attrList = graph_kwargs['attrList']

	layers = opt.layers
	print('attrlist and layers: ', attrList, layers)

	loss_values = train(opt.model, g, graph_inputs, output_dir,
						attrList,
						layers=layers,
						save_freq=opt.model_save_freq,
						trainEmbed=opt.trainEmbed,
						updateGAN=opt.updateGAN,
						opt=opt
						)
	loss_values = np.array(loss_values)

	np.save('./{}/loss_values.npy'.format(output_dir), loss_values)
	f, ax = plt.subplots(figsize=(10, 4))
	ax.plot(loss_values)
	f.savefig('./{}/loss_values.png'.format(output_dir))
