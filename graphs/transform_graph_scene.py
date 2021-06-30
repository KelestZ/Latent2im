import importlib
import numpy as np
import math

def get_transform_graphs(model):

	print("Getting semantic transform graphs for {} model...".format(model))
	print('Load transform_base in get_transform_graphs')
	transform_base_name = 'graphs.' + model + '.transform_base'
	base = importlib.import_module(transform_base_name)

	transform_op_name = 'graphs.' + model + '.transform_op'
	print('Load transform op in get_transform_graphs: %s' % transform_op_name)
	op = importlib.import_module(transform_op_name)

	print('Load constants in get_transform_graphs')
	constants_name = 'graphs.' + model + '.constants'
	constants = importlib.import_module(constants_name)

	print(transform_op_name, constants_name)

	class SceneGraph(base.PixelTransform, op.SceneTransform):
		def __init__(self, lr=0.001, walk_type='NNz', loss='l2', eps=1.41, N_f=4,
					 **kwargs):
			nsliders = 1
			self.walk_type = walk_type
			self.num_channels = constants.NUM_CHANNELS
			self.Nsliders = nsliders
			self.img_size = constants.resolution

			base.PixelTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
			op.SceneTransform.__init__(self)

		def vis_image_batch(self, graph_inputs, filename,
							batch_start, wgt=False, wmask=False,
							num_panels=7, max_alpha=None, min_alpha=None, N_attr=40):

			zs_batch = graph_inputs['z']

			if max_alpha is not None and min_alpha is not None:
				alphas = np.linspace(min_alpha, max_alpha, num_panels)
			else:
				# alphas = self.vis_alphas(self.num_panels)
				alphas = np.linspace(0, 1, num_panels)
			alphas_to_graph = []
			alphas_to_target = []

			for a in alphas:
				slider = self.scale_test_alpha_for_graph(a, zs_batch)
				alphas_to_graph.append(slider)
				alphas_to_target.append(a)
			return alphas_to_graph, alphas_to_target

	class faceGraph(base.PixelTransform, op.FaceTransform):
		def __init__(self, lr=0.001, walk_type='NNz', loss='l2', eps=1.41, N_f=4,
					 **kwargs):
			nsliders = 1
			self.walk_type = walk_type
			self.num_channels = constants.NUM_CHANNELS
			self.Nsliders = nsliders
			self.img_size = constants.resolution

			base.PixelTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
			op.FaceTransform.__init__(self)

		def vis_image_batch(self, graph_inputs, filename,
							batch_start, wgt=False, wmask=False,
							num_panels=7, max_alpha=None, min_alpha=None, N_attr=40):

			zs_batch = graph_inputs['z']

			####
			if max_alpha is not None and min_alpha is not None:
				alphas = np.linspace(min_alpha, max_alpha, num_panels)
			else:
				alphas = np.linspace(0, 1, num_panels)

			alphas_to_graph = []
			alphas_to_target = []
			for a in alphas:
				#new_label = np.array([x, y])
				new_label = a
				slider = self.scale_test_alpha_for_graph(new_label, zs_batch)
				alphas_to_graph.append(slider)
				alphas_to_target.append(new_label)
			return alphas_to_graph, alphas_to_target

	# class xrayGraph(base.PixelTransform, op.XrayTransform):
	# 	def __init__(self, lr=0.001, walk_type='NNz', loss='l2', eps=1.41, N_f=4, **kwargs):
	# 		nsliders = 1
	# 		self.walk_type = walk_type
	# 		self.num_channels = constants.NUM_CHANNELS
	# 		self.Nsliders = nsliders
	# 		self.img_size = constants.resolution
	#
	# 		base.PixelTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
	# 		op.XrayTransform.__init__(self)
	#
	# 	def vis_image_batch(self, graph_inputs, filename,
	# 						batch_start, wgt=False, wmask=False,
	# 						num_panels=7, max_alpha=None, min_alpha=None, N_attr=40):
	#
	# 		zs_batch = graph_inputs['z']
	#
	# 		####
	# 		if max_alpha is not None and min_alpha is not None:
	# 			alphas = np.linspace(min_alpha, max_alpha, num_panels)
	# 		else:
	# 			alphas = np.linspace(0, 1, num_panels)
	#
	# 		alphas_to_graph = []
	# 		alphas_to_target = []
	# 		for a in alphas:
	# 			#new_label = np.array([x, y])
	# 			new_label = a
	# 			slider = self.scale_test_alpha_for_graph(new_label, zs_batch)
	# 			alphas_to_graph.append(slider)
	# 			alphas_to_target.append(new_label)
	# 		return alphas_to_graph, alphas_to_target

	# graphs = [SceneGraph, faceGraph, xrayGraph]

	graphs = [SceneGraph, faceGraph]

	return graphs

