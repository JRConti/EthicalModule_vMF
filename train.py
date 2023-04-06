"""
python3 train.py -p ../SSD/JR/Arcface_MS1MV3_R100_idemia_0_33/ -d 512 -g 0 -s k,k

Jean-Remy Conti
2022
"""


import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import datetime
from tqdm import tqdm
from pathlib import Path

from utils.numpy_dataset import NumpyDataset
from backbones import get_model
from ethical_module import MLP, EthicalModule, FairModel


parser = argparse.ArgumentParser(description='vMF training')
parser.add_argument('-p', '--path_data', help='Path to numpy data folder', type=str)
parser.add_argument('-d', '--dim', type=int, default=512, help='Feature space dimension') # 512 
parser.add_argument('-s', '--scales', help='list of scale parameters', type=str)
parser.add_argument('-n', '--network', default=None, help='Backbone network (r18, r34, r50, r100) of pre-trained model. '+
												' Set to None to avoid validation during training.', type=str)
parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU ids', nargs='+')


#------------ Parameters ---------------#

seed = 42 

n_epochs = 50
n_epochs_to_print = 1 # Number of epochs between each epoch loss print
n_epochs_to_save = 2 # Number of epochs between each model checkpoint save

lr =  0.01
batch_size = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


def train(dataset, feat_dim = 512, kappas = 5.0, n_epochs = 200, lr = 0.01, seed = 42, batch_size = 32, device_ids= None):
	"""
	Parameters
	----------
	dataset: NumpyDataset instance imported from numpy_dataset.
		dataset[input_idx] : tuple (embeddings[idx], torch.tensor(label[idx], gender[idx]))
		embeddings = dataset[:][0] 
		embeddings[idx] = dataset[idx][0]
		torch.tensor(label, gender) = dataset[:][1] || shape (n_points x 2)
		label = dataset[:][1][:,0]
		gender = dataset[:][1][:,1]
		label[idx] = dataset[idx][1][0]
		gender[idx] = dataset[idx][1][1]
	feat_dim: int
	kappas: float or list of floats
		See vmf_loss.py.
	n_epochs: int
	lr: float
	seed: int
		Used to reproduce some random operations such as shuffling dataset.
	batch_size: int
	device_ids: list
		List of GPU ids to use torch.nn.DataParallel.
		Can be None if only one device is used.
	"""

	torch.manual_seed(seed)

	n_points = len(dataset)
	n_classes = torch.max(dataset[:][1][:,0]).item() + 1
	assert n_classes <= n_points, 'Please use more data points than centroids.'
	
	# Reduce list of concentration parameters
	if type(kappas) != list:
		kappas = [kappas]
	n_kappas = torch.unique( torch.FloatTensor(kappas) ).size(0)

	# Used to define vMF loss
	if n_kappas > 1:
		# Get the ordered set of unique labels alongside their corresponding genders
		labels_set = dataset.get_labels_set().to(device)
	else:
		kappas = kappas[0]
		labels_set=None

	# Display info of current model
	print('#-------------- INFO --------------#')
	print('Number of classes: ', n_classes)
	print('Number of data points: ', n_points, '\n')
	print('Concentration parameters: ', kappas, '\n')
	print()

	# ------- Initialize training ------- # 
	train_size = len(dataset)
	train_loader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=True)
	
	# Define model
	d_in = dataset[0][0].size()[0]
	h = 2 * d_in 																
	model = MLP(d_in, h, feat_dim) # network transforming embeddings
	module = EthicalModule(model, n_classes, kappas, labels_set= labels_set)
	module.to(device)
	if type(device_ids) == list and len(device_ids) > 1:
		print('multi-GPU')
		# multi-GPU setting
		module = torch.nn.DataParallel(module, device_ids = device_ids).to(device)
		# Save model architecture (but random weights)
		torch.save(module.module.model, working_dir + 'model_dummy.pt')
	else:
		torch.save(module.model, working_dir + 'model_dummy.pt')

	optimizer = torch.optim.Adam(module.parameters(), lr=lr)
	
	train_losses = []
	print('#------------ TRAINING ------------#')
	n_iter = 1
	for epoch in tqdm(range(n_epochs)):

		#------------ TRAINING ------------#
		module.train()
		# Init loss of current epoch
		train_loss = 0.0

		# Batch process
		pbar = tqdm(total= train_size // batch_size) 
		for data, targets in train_loader:

			data, targets = data.to(device), targets.to(device)

			# Forward
			loss, _ = module(data, forward = True, return_loss = True, labels = targets[:,0], get_features = False) 
			# multi-GPU
			loss = loss.mean()			
			assert np.isnan(loss.item()) == False, "Loss is NaN !"

			# Backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Update loss of current epoch
			train_loss += loss.item() * float(len(data)) / float(train_size) 
			tb.add_scalar('loss_batch', loss.item(), n_iter)		
			pbar.update(1) 
			n_iter += 1
		pbar.close() 
		train_losses.append(train_loss)	

		tb.add_scalar('loss', train_loss, epoch+1)
		tb.flush()

		# Print loss
		if (epoch+1) % n_epochs_to_print == 0 or (epoch == n_epochs -1):
			print('Epoch [{}/{}], Train loss: {:.10f}' 
				.format(epoch+1, n_epochs, train_loss))

		# Save model at each epoch 
		if (epoch+1) % n_epochs_to_save == 0:
			if type(device_ids) == list and len(device_ids) > 1:
				torch.save(module.module.model.state_dict(), working_dir + 'checkpoints/' + str(epoch+1) + '.pt' )	
			else:
				torch.save(module.model.state_dict(), working_dir + 'checkpoints/' + str(epoch+1) + '.pt' )	
	tb.close()

	# Save entire model
	if type(device_ids) == list and len(device_ids) > 1:
		torch.save(module.module.model, working_dir + 'model.pt')
	else:
		torch.save(module.model, working_dir + 'model.pt')
	os.remove(working_dir + 'model_dummy.pt')

	# Save evolution of losses
	np.save(working_dir + 'train_loss', np.array(train_losses))

	# Save hyperparameters to config file
	to_write = []
	to_write.append(str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")) + '\n\n')
	to_write.append('Number of epochs: ' + str(epoch+1) + '\n')
	to_write.append('Seed: ' + str(seed) + '\n\n')

	to_write.append('H: ' + str(h) + '\n')
	to_write.append('Feature dimension: ' + str(feat_dim) + '\n\n')

	to_write.append('Learning Rate: ' + str(lr) + '\n')
	to_write.append('Batch Size: ' + str(batch_size) + '\n\n')

	kappas_str = str(kappas)[1:-1] if str(kappas)[0] == '(' else str(kappas)
	to_write.append('Kappa : ' + kappas_str  + '\n\n')

	with open(working_dir + 'config_training.txt', mode = 'w') as f_w:
			for line in to_write:
				f_w.write(line)

	print('------------------------------------\n')




if __name__ == "__main__":
	
	# Take info
	args = parser.parse_args()
	print("args", args, '\n')

	# Load dataset
	dataset = NumpyDataset(data_folder = args.path_data, root_img = None, gender = True)

	# Concentration parameters
	kappas = [float(item) for item in args.scales.split(',')]
	kappa_m = kappas[0]
	kappa_f = kappas[0] if len(kappas) == 1 else kappas[1]

	# Output training files destination (USED IN TRAIN FUNCTION)
	working_dir = args.path_data + 'training/kappaM_' + str(int(kappa_m)) + '_kappaF_' + str(int(kappa_f)) + '/' 
	Path(working_dir + 'checkpoints/').mkdir(parents= True, exist_ok= True)
	# Tensorboard
	tb = SummaryWriter(working_dir + 'logs')

	# Start training
	start = time.time()
	train(dataset, feat_dim = args.dim, kappas = kappas, n_epochs = n_epochs, 
		 lr = lr, seed = seed, batch_size = batch_size, device_ids = args.gpu)
	end = time.time()
	print('%.2f s' % (end-start))




