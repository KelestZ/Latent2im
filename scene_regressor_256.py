import os
from shutil import copyfile
import torch
import glob
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# 6826/1745 = 8571

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from math import ceil
from  torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import average_precision_score

'''
python res50_regressor_256.py
'''

class CustomDataset(Dataset):
	def __init__(self, folder_path, label_dict, split_file, image_size=256):
		self.label_dict = label_dict
		self.image_size = image_size
		self.split = []
		with open(split_file, 'r') as f:
			for i in f.readlines():
				self.split.append(i.strip())

		self.transform = transforms.Compose([
				        transforms.Resize(self.image_size),
				        transforms.CenterCrop(self.image_size),
				        transforms.ToTensor(),
				        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				    ])

		self.total_list = glob.glob(folder_path+'/*/*')
		self.image_list = []

		for i in self.total_list:
			if '/'.join(i.split('/')[-2:]) in self.split:
				self.image_list.append(i)
		# Calculate len
		self.data_len = len(self.image_list)

	def __getitem__(self, index):
		single_image_path = self.image_list[index]

		im_as_im = Image.open(single_image_path)

		# Transform image to tensor, change data type
		im_as_ten = self.transform(im_as_im)
		# Get label(class) of the image based on the file name
		label_idx = '/'.join(self.image_list[index].split('/')[-2:])

		label = torch.Tensor(self.label_dict[label_idx])
		return (im_as_ten.cuda(), label.cuda())

	def __len__(self):
		return self.data_len


import csv
def load_labelfile(path):
	labels = {}
	with open(path, 'r') as csvfile:
		lines = csv.reader(csvfile, delimiter='\t')
		for line in lines:
			labels[line[0]] =  np.array([ float(i.split(',')[0]) for i in line[1:]])
	return labels

def load_ckpt(path, model, optimizer):
	# ckpt = '/home/peiye/ImageEditing/scene_regressor/checkpoint/100_dict.model'
	
	ckpt = torch.load(path)
	model.load_state_dict(ckpt['model'])
	optimizer.load_state_dict(ckpt['optm'])
	return model, optimizer

if __name__ == '__main__':
	if not os.path.exists('./checkpoint_256'):
		os.mkdir('./checkpoint_256')
		os.mkdir('./log_256')

	# Complete the data path
	path = '/transient_scene/imageAlignedLD/'
	label_path = '/transient_scene/annotations/annotations.tsv'
	split_path = '/transient_scene/training_test_splits/random_split/'


	label_file = load_labelfile(label_path)

	train_data = CustomDataset(path, label_file, 
								split_file=split_path+'training.txt',
								image_size=256)

	test_data = CustomDataset(path, label_file, 
								split_file=split_path+'test.txt',
								image_size=256)

	trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
	testloader = DataLoader(test_data, batch_size=32, shuffle=False)


	model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
	model.fc = torch.nn.Linear(2048, 40)
	model = model.cuda()
	
	N_EPOCH = 500
	N_ITER = len(trainloader)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	# model, optimizer = load_ckpt(path='/home/peiye/ImageEditing/scene_regressor/checkpoint/200_dict.model', 
	# 							 model=model, 
	# 							 optimizer=optimizer)

	writer = SummaryWriter(log_dir='./log_256'+'/')

	criterion = nn.MSELoss().cuda()
	print('EPOCH: ', N_EPOCH, 'ITER: ', N_ITER)

	pbar = tqdm(range(N_EPOCH))

	for epoch in pbar:
		it = iter(trainloader)
		for iter_ in range(N_ITER):
			data, label = next(it) # [32, 3, 256, 256]
			preds = model(data)
			optimizer.zero_grad()
			Loss = criterion(preds, label).mean()
			Loss.backward()
			optimizer.step()

			pbar.set_description(f'Iter {iter_ + 1} Loss: {Loss:.5f}')
			if (N_ITER* epoch + iter_) % 50 == 0:
				writer.add_scalar('Train/Loss', Loss, N_ITER*epoch + iter_)

		# if epoch % 5 == 0 and epoch != 0:

		if epoch % 1 == 0 and epoch != 0:
			# Test
			with torch.no_grad():	
				test_loss = []
				# aps = []

				for test_data, test_label in testloader:
					test_preds = model(test_data)
					test_loss.append(criterion(test_preds, test_label).mean().cpu().item())

					# print(test_preds.cpu().numpy().shape,test_label.cpu().numpy().shape)
					# ap = average_precision_score(test_preds.cpu().numpy(),test_label.cpu().numpy())
					# aps.append(ap)
					# break

				pbar.set_description(f'Test epoch {epoch}; Loss: {np.mean(test_loss):.5f}')
				writer.add_scalar('Test/MSE', np.mean(test_loss), epoch)
				# writer.add_scalar('Test/AP', np.mean(aps), epoch)
				

		# Save model
		torch.save({
              'model': model.state_dict(),
              'optm': optimizer.state_dict()},
               'checkpoint_256' + f'/{str(epoch + 1).zfill(3)}_dict.model')
	writer.close()


