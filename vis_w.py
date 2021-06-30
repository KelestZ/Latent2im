import os
import numpy as np
from options.vis_options import VisOptions
from utils import util
import importlib
import graphs
from utils import html

import torch
import torchvision
import torchvision.transforms as transforms

"""
Usage example:

python vis_w.py models_celeba/stylegan_v2_real_face_linear_lr0.0001_l2_w/opt.yml \
        --gpu 3 --noise_seed 0 --num_samples 30 --num_panels 10 \
        --save_path_w  ./models_celeba/stylegan_v2_real_face_linear_lr0.0001_l2_w/model_w_10_final_walk_module.ckpt           
"""

if __name__ == '__main__':
	v = VisOptions()
	v.parser.add_argument('--num_samples', type=int, default=10,
						  help='number of samples per category')
	v.parser.add_argument('--num_panels', type=int, default=7,
						  help='number of panels to show')
	v.parser.add_argument('--max_alpha', type=float, default=1,
						  help='maximum alpha value')
	v.parser.add_argument('--min_alpha', type=float, default=0,
						  help='minimum alpha value')
	v.parser.add_argument('--layers', type=str, default=None,
						  help='minimum alpha value')



	v.parser.add_argument('--trainEmbed', action='store_true')
	v.parser.add_argument('--updateGAN', action='store_true')

	opt, conf = v.parse()
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

	if opt.output_dir:
		output_dir = opt.output_dir
	else:
		output_dir = os.path.join(conf.output_dir, 'images')
	os.makedirs(output_dir, exist_ok=True)

	graph_kwargs = util.set_graph_kwargs(conf)
	print('Load utils and constants: %s' % conf.model)
	graph_util = importlib.import_module('graphs.' + conf.model + '.graph_util')
	constants = importlib.import_module('graphs.' + conf.model + '.constants')


	print('Find_model_using_name')
	model = graphs.find_model_using_name(conf.model, conf.transform)

	print('Model initialization')
	g = model(**graph_kwargs)
	print('Load multi models')
	if opt.updateGAN:
		g.load_multi_models(opt.save_path_w, opt.save_path_gan, trainEmbed=opt.trainEmbed, updateGAN=opt.updateGAN)
	else:
		g.load_multi_models(opt.save_path_w, None, trainEmbed=opt.trainEmbed, updateGAN=opt.updateGAN)

	num_samples = opt.num_samples
	noise_seed = opt.noise_seed
	batch_size = constants.BATCH_SIZE

	graph_inputs = graph_util.graph_input(g, num_samples, seed=noise_seed)

	print('Start visualization')

	if 'final' in opt.save_path_w:
		epochs = opt.save_path_w.split('/')[-1].split('_')[2]
	else:
		epochs = opt.save_path_w.split('/')[-1].split('_')[2]

	filename = os.path.join(output_dir, 'w_{}_seed{}'.format(
		epochs, noise_seed))

	name = conf.attrList.strip().split(',')[0]
	print('Name and epochs: ', name, epochs)

	if opt.layers == 'None' or opt.layers == None:
		layers = None
	else:
		layers = [int(i) for i in opt.layers.split(',')]

	# 256*256 resolution
	step = 6

	for batch_start in range(0, num_samples, batch_size):
		s = slice(batch_start, min(num_samples, batch_start + batch_size))
		graph_inputs_batch = util.batch_input(graph_inputs, s)

		max_alpha = opt.max_alpha
		min_alpha = opt.min_alpha

		new_filename = filename + '_{}_max{}_min{}'.format(name, max_alpha, min_alpha)

		alphas_to_graph, alphas_to_target = g.vis_image_batch(graph_inputs_batch, new_filename, s.start,
																  num_panels=opt.num_panels, max_alpha=max_alpha,
																  min_alpha=min_alpha, wgt=True)

		g.vis_multi_image_batch_alphas(graph_inputs_batch, new_filename,
									   alphas_to_graph=alphas_to_graph,
									   alphas_to_target=alphas_to_target,
									   layers=layers,
									   batch_start=s.start,
									   name=name, wgt=False, wmask=False,
									   trainEmbed=opt.trainEmbed, computeL2=False,
									   given_w=None)

	html.make_html(output_dir)




