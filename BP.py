"""
Usage:
python BP.py --batch_size 1 --optim Adam --dataset scene \
--n_loops 4000 --path ./data/scene --save_path ./results

python BP.py --batch_size 1 --optim Adam --dataset ffhq \
--n_loops 4000 --path ./data/face --save_path ./results_face

"""
import os
import argparse
import itertools
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.autograd import grad as torchgrad
from torchvision.utils import make_grid
from torchvision import transforms, utils
from pprint import pprint
import math
import torch.nn as nn
# from stylegan_v2 import stylegan2 as stylegan
from graphs.stylegan_v2_real import stylegan2 as stylegan
from perceptual_vgg.vgg import Vgg16



parser = argparse.ArgumentParser(description='Backprop')
parser.add_argument('--latent_dim', type=int, default=512, metavar='D',
					help='number of latent variables (default: 2)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
					help='number of examples to eval at once (default: 10)')
parser.add_argument('--n-batch', type=int, default=1, metavar='B',
					help='number of batches to eval in total (default: 10)')
parser.add_argument('--chain-length', type=int, default=500, metavar='L',
					help='length of ais chain (default: 500)')
parser.add_argument('--ckpt_path', type=str, default='./mnist_generator.pth',  # './G_80000.pth',#
					metavar='C', help='path to checkpoint')
parser.add_argument('--num_samples', type=int, default=9,
					metavar='I', help='number of importance samples')
parser.add_argument('--not_use_gpu', action='store_true',
					help='use gpu or not')
parser.add_argument('--gpu', type=str, default='0',
					metavar='G', help='assign gpu device')
parser.add_argument('--n_loops', type=int, default=500,
					metavar='T', help='number of temperature schedule')
parser.add_argument('--resolution', type=int, default=256, choices=[128, 256, 512],
					help="image resolution")
parser.add_argument('--block', action='store_true',
					help="inpaining or not")
parser.add_argument('--optimizer', type=str, choices=['Adam', 'GD'],
					help="optimizer")
####### New ones
parser.add_argument('--dataset', type=str, choices=['ffhq', 'scene', 'anime'])
parser.add_argument('--path', type=str)
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
torch.set_num_threads(1)

pprint(args)


def gram(x):
	(bs, ch, h, w) = x.size()
	f = x.view(bs, ch, w*h)
	f_T = f.transpose(1, 2)
	G = f.bmm(f_T) / (ch * h * w)
	return G

def log_likelihood(logit, target, mask=None, sigma=0.25):
	"""
	:param logit: [batch_size, 1, img_size, img_size]
	:param target:  [batch_size, 1, img_size, img_size]
	:param mask:  [1,1,img_size, img_size]
	:param sigma:
	:return:
	"""
	# print('check size in log_likelihood: ', logit.size(), target.size(), mask.size())
	diff = (logit - target)
	distri = 'berboulli'
	if distri == 'berboulli':
		return - torch.sum(diff.pow(2), [1, 2, 3])
	# return - torch.mean(diff.pow(2), [1, 2, 3])
	# return -torch.sum(torch.mul(diff, diff), 1)

	elif distri == 'gaussian':
		k = logit.size(2) * logit.size(3)
		# loss = - torch.sum(diff.pow(2), [-1, -2, -3]) / (2 * sigma) - 0.5 * k * np.log(2 * np.pi) - 0.5 * k * np.log(sigma)
		# k = 2
		loss = - torch.sum(diff.pow(2), [1, 2, 3]) / (2 * sigma) - 0.5 * k * np.log(2 * np.pi) - 0.5 * k * np.log(sigma)

		return loss

class Trainer(object):
	def __init__(self, model,latent_dim=512,
				block=True, resolution=256,
			  n_loops=500, optim='Adam', lr=1e-4,
			  show_freq=500):
		self.model = model
		self.latent_dim = latent_dim
		self.resolution = resolution
		self.lr = lr
		self.show_freq = show_freq
		self.n_loops = n_loops
		self.MSE_loss = nn.MSELoss()
		with torch.no_grad():
			self.mean_latent = self.model.netG.mean_latent(4096)
		self.model.netG.cuda()
		dtype = torch.cuda.FloatTensor
		self.vgg = Vgg16().type(dtype)

		self.optim = optim
	def train(self, loader, save_img_dir):
		it = iter(loader)
		mses = []

		for i in range(len(loader)):
			print(len(loader), i)
			batch, _ = next(it)

			visual_imgs(batch, save_path=save_img_dir + '/org_%d.png' % i)
			# get w
			batch = batch.cuda()
			current_w = self.mean_latent		# 1, 512
			#w = current_w.view(batch.size(0), 1, self.latent_dim)
			# w = w.repeat(1, int((math.log2(self.resolution / 4) + 1) * 2), 1).requires_grad_()
			multi_w = [current_w] * 14
			w = torch.stack(multi_w).transpose(0, 1)

			w = w.repeat(batch.size(0), 1, 1) # N, 14, 512
			w.requires_grad_()
			if self.optim == 'Adam':
				self.optimizer = torch.optim.Adam([w], lr=self.lr, betas=(0.5, 0.9))
			elif self.optim == 'GD':
				self.optimizer = torch.optim.SGD([w], lr=self.lr, momentum=0.9)

			for loop in tqdm(range(1, self.n_loops + 1, 1)):
				# Get logit
				output, _ = self.model.netG(w, input_is_latent=True)
				# Compute loss
				# l2 = self.l2_loss(batch, output).mean()
				N = torch.Tensor([batch.size(2) * batch.size(3) * 3]).cuda()
				nllloss = -log_likelihood(output, batch)/N

				p_loss = self.perceptual_loss(batch, output).mean()
				# loss = l2 + p_loss
				loss = (nllloss + p_loss).sum()

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				mse = loss.item()
				mses.append(mse)
				if loop % self.show_freq == 0 or loop == self.n_loops:
					print('[%d, %d]: ' % (loop, self.n_loops), 'MSE : %.3f' % mse, 'w: ', w[0, 0, :4])

					visual_imgs(output,
								plot_title='%d iterations' % loop,
								is_show=False,
								is_save=True,
								save_path=save_img_dir + '/%d_iterations_%d.png' % (i, loop))

			visual_imgs(output, save_path=save_img_dir + '/%d_final.png' % (i))

			plot_loss(mses, is_show=False, is_save=True, save_path=save_img_dir, name=i)
			self.save_w(w, save_path=save_img_dir + '/latent/', name='%d_w.npy' % i)

	def perceptual_loss(self, batch, logit):
		data_features = self.vgg(batch)
		logit_features = self.vgg(logit)
		data_feature_gram = [gram(fmap) for fmap in data_features]
		logit_feature_gram = [gram(fmap) for fmap in logit_features]

		p_loss_vgg = torch.zeros(batch.size(0)).cuda()
		for idx in range(4):
			N = torch.Tensor([data_feature_gram[idx].size(1) * data_feature_gram[idx].size(2)]).cuda()
			diff = (data_feature_gram[idx] - logit_feature_gram[idx]).cuda()
			p_loss_vgg += torch.sum(diff.pow(2), [1, 2]) * N
		return p_loss_vgg

	def l2_loss(self, logits, gt):
		loss = self.MSE_loss(logits, gt)
		return loss

	def save_w(self, w, save_path, name):
		if not os.path.exists(save_path):
			os.makedirs(save_path, exist_ok=True)
		np.save(save_path + name, w.detach().cpu().numpy())

# def back_prop(model, dataset, n_sample, use_gpu,
# 			  atent_dim=512, block=True, resolution=128,
# 			  n_loops=500, optim='Adam', lr=1e-4,
# 			  show_freq=100):
# 	"""
# 	:param model:
# 	:param loader:
# 	:param n_sample:
# 	:param use_gpu:
# 	:param optim:
# 	:return:
# 	"""
# 	# Fix random seed
# 	torch.manual_seed(1234)
# 	np.random.seed(1234)
#
# 	mses = []
# 	for i, batch in enumerate(dataset):
# 		B = batch.size(0) * n_sample  # batch_size, n_sample  1*64
# 		batch = utils.safe_repeat(batch, n_sample)  # (batch_size, 1, 32, 32)
# 		batch = batch.cuda()
# 		current_z = torch.randn([B, latent_dim]).cuda()
# 		# get w
# 		current_w = model.netG.style(current_z)
# 		multi_w = [current_w] * (math.log(resolution/4) + 1) * 2
#
# 		# get image
# 		w = torch.stack(multi_w).transpose(0, 1)
# 		w.requires_grad_()
#
# 		output, _ = model.netG(w, input_is_latent=True)
#
# 		if not os.path.exists(save_img_dir):
# 			print('[INFO] Make the saving dir %s... \n' % save_img_dir)
# 			os.mkdir(save_img_dir)
# 		sub_saving_dir = '%s_%d/' % (optim, n_loops)
# 		save_img_dir = save_img_dir + sub_saving_dir
# 		if not os.path.exists(save_img_dir):
# 			os.mkdir(save_img_dir)
#
# 		visual_imgs(result, plot_title='Before loop', is_show=False, is_save=True,
# 					save_path=save_img_dir + 'Before_loop.png')
#
# 		visual_imgs(batch[0] * mask,
# 					plot_title='real iterations',
# 					is_show=False,
# 					is_save=True,
# 					save_path=save_img_dir + 'real_.png')
#
# 		def l2_loss(z):
# 			diff = ((model.netG(z) - batch) * mask).cuda()
# 			loss = torch.sum(diff.pow(2), [-1, -2, -3]).cuda()
# 			return loss
#
# 		def grad_GAN(z):
# 			# grad w.r.t. outputs; mandatory in this case
# 			if use_gpu:
# 				grad_outputs = torch.ones(B).cuda()
# 			else:
# 				grad_outputs = torch.ones(B)
#
# 			grad = torchgrad(l2_loss(z), z, grad_outputs=grad_outputs)[0]
# 			# clip by norm
# 			# max_ = B * latent_dim * 100.
# 			# grad = torch.clamp(grad, -max_, max_)
# 			grad.requires_grad_()
# 			return grad
#
# 		for loop in tqdm(range(1, n_loops + 1, 1)):
# 			l2 = l2_loss(current_z)
#
# 			# current_z = current_z - lr * grad_GAN(current_z)
# 			optimizer.zero_grad()
# 			l2.backward()
# 			optimizer.step()
# 			mse = l2.item() #utils.mean_squared_error(result, batch, mask)
# 			print('MSE : %.3f' % mse)
# 			mses.append(mse)
# 			if loop % show_freq == 0 or loop == n_loops:
# 				result = model.netG(current_z)  # (batch_size, 1, 32, 32)
# 				print('[%d, %d]: ' % (loop, n_loops))
# 				visual_imgs(result,
# 							plot_title='%d iterations' % loop,
# 							is_show=False,
# 							is_save=True,
# 							save_path=save_img_dir + 'iterations_%d.png' % loop)
#
# 		plot_loss(mses, is_show=False, is_save=True, save_path=save_img_dir)



def plot_loss(loss, is_show=False, is_save=False, save_path=None,name=None):
	x = [i for i in range(len(loss))]
	plt.cla()
	plt.plot(x, loss)
	np.save(save_path + '/loss_back.npy', np.array(loss))

	plt.xlabel('Loops')
	plt.ylabel('L2 loss')
	if is_show:
		plt.show()
	if is_save:
		plt.savefig(save_path + '/%s_loss.png' % name)

def visual_imgs(result, plot_title=None, is_show=False, is_save=False, save_path=None):
	try:
		with torch.no_grad():
			utils.save_image(
				result,
				save_path,
				nrow=1,
				normalize=True,
				range=(-1, 1))
	except:
		plt.cla()
		result_grid = make_grid(result, nrow=1, normalize=True)  # (3, 274, 274)

		img = transforms.ToPILImage()(result_grid.cpu())
		img.save(save_path)

def get_stylegan2_module(dataset, lr=1e-4):

	if dataset == 'scene':
		# Scene
		base_dir = '/home/peiye/ImageEditing/stylegan2-pytorch/checkpoint/'
		ckpt = torch.load(base_dir + '190000.pt')

	elif dataset == 'ffhq':
		base_dir = '/shared/rsaas/zpy/2nd_year/stylegan2_celeba/pretrained_ffhq/'
		ckpt = torch.load(base_dir + '550000.pt')

	module = stylegan.StyleGAN(lr=lr)
	module.netG.load_state_dict(ckpt['g_ema'],strict=False)
	module.netD.load_state_dict(ckpt['d'])
	module.optimizerG.load_state_dict(ckpt['g_optim'])
	module.optimizerD.load_state_dict(ckpt['d_optim'])
	print('Finish loading the pretrained model')
	return module


def main():
	# Load model
	#	transforms.Resize(args.resolution),
	model = get_stylegan2_module(args.dataset)
	transform = transforms.Compose(
		[	transforms.Resize(args.resolution),
			transforms.CenterCrop(args.resolution),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
		]
	)
	dataset = torchvision.datasets.ImageFolder(args.path, transform=transform)

	loader = torch.utils.data.DataLoader(dataset,
										 batch_size=args.batch_size,
										 )

	train = Trainer(model=model, latent_dim=args.latent_dim,
					resolution=args.resolution,
					n_loops=args.n_loops, optim=args.optimizer,
					lr=args.lr)

	train.train(loader, save_img_dir=args.save_path)

	# # run back propagation
	# back_prop(
	# 	model,
	# 	loader,
	# 	n_sample=args.num_samples,
	# 	use_gpu=args.use_gpu,
	# 	latent_dim=args.latent_dim,
	# 	resolution=args.resolution,
	# 	block=args.block,
	# 	n_loops=args.n_loops,
	# 	optim=args.optimizer)

if __name__ == '__main__':
	main()
