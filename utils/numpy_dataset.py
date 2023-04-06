"""
Jean-Remy Conti
2022
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


class NumpyDataset(torch.utils.data.Dataset):
	'''
	A class to read npy files containing face embeddings and labels, as
	well as img_list txt file. The storage of data is very efficient by 
	using numpy memmaps. 
	'''
	def __init__(self, data_folder, root_img = None, gender = False):
		'''
		
		Parameters
		----------
		data_folder: str
			Path to folder containing face embeddings/labels npy files,
			as well as img_list txt file.
		root_img: str
			Path to root folder containing images of the dataset.
		gender: boolean
			Set to True to load gender labels, located at data_folder.
			Here, gender values are only binary: 0 or 1.
		'''

		# Keep track of this option
		self.has_gender = gender
		if data_folder[-1] != '/':
			data_folder += '/'
		self.data_folder = data_folder
		self.root_img = root_img

		print('Loading data ...')

		# Load labels (use of memmap for memory efficiency)
		self.labels = np.load(data_folder + 'labels.npy', 
							mmap_mode = 'r').copy() 
		# -> type: np.memmap

		# Check that labels are: 0, 1, ..., max_label
		self.n_labels = np.amax(self.labels) + 1
		assert set(self.labels) == set(range(self.n_labels))

		# Load face embeddings -> type: np.memmap
		self.embeddings = np.load(data_folder + 'embeddings.npy', 
								mmap_mode = 'r').copy() 
		assert self.embeddings.shape[0] == self.labels.shape[0], (
			'Input data and corresponding labels have to share 1 dimension')

		# Load genders
		if self.has_gender:
			self.genders = np.load(data_folder + 'genders.npy', 
									mmap_mode = 'r').copy()
			assert self.genders.shape[0] == self.embeddings.shape[0], (
			'Input data and corresponding genders have to share 1 dimension')
			assert set(self.genders) == set([0,1])

	def __len__(self):

		return self.labels.shape[0]

	def __getitem__(self, idx):
		'''
		idx	-> np.memmap access to data for memory efficiency
			-> wrap into torch tensors

		Returns
		-------
		(data, labels): tuple
		data: tensor shape (len(idx), input_dimension)
		labels: tensor shape (len(idx),) 
				or (len(idx),2) if self.has_gender
		labels[idx] or labels[idx][0] if self.has_gender: 
			0 <= int < n_classes
			Identity label of self.embeddings[idx].
		labels[idx][1] if self.has_gender: 0 or 1
			Binary gender of self.embeddings[idx].
		'''

		# Input
		data = self.embeddings[idx]
		data = torch.tensor(data).float()				

		# Corresponding class (np array format)
		label = self.labels[idx]
		if type(label) is not np.ndarray:
			label = np.array([label])
		
		if self.has_gender:
			# Corresponding gender (np array format)
			gender = self.genders[idx]
			if type(gender) is not np.ndarray:
				gender = np.array([gender])

			# Concatenate genders with labels -> shape (n_points x 2)
			labels = np.array([label, gender])
			labels = torch.from_numpy(labels).long().transpose(0,1)   
		else:
			labels = torch.tensor(label).long()
		
		# Squeeze in case of a single idx as input ; otherwise, output
		# is:    torch.tensor( [[ labels[idx][0], labels[idx][1] ]] )    
		# No need for that for slices but inefficient in that case.
		return (data, labels.squeeze(0))

	def get_labels_set(self):
		'''
		Get the ordered set of unique labels (0 -> self.n_labels -1),
		alongside their corresponding genders (if self.has_gender).

		Returns
		-------
		torch tensor (self.n_labels, 2)
		* column 1: unique labels (0 -> self.n_labels -1)
		* column 2: corresponding genders
		'''

		if self.has_gender:

			# Find unique labels and corresponding genders
			print('\nSummarizing labels and genders ...')
			
			# Find genders of the set of labels
			labels_set_genders = [] 
			for label in tqdm(range(self.n_labels)):

				# Find genders of label
				current_genders = list(set(
								self.genders[ self.labels == label ]))
				assert len(current_genders) == 1, ('Label {} has following ' +
					'genders: {}'.format(label, current_genders))

				labels_set_genders.append( current_genders[0] )
			
			# Format each attribute
			labels_set = torch.arange(self.n_labels).long().unsqueeze(1)
			labels_set_genders = (
				torch.tensor(labels_set_genders).long().unsqueeze(1))
			
			# (N x 2)
			return torch.cat([ labels_set, labels_set_genders ], dim= 1) 

		else:
			# (N,)
			print('no gender')
			return torch.arange(self.n_labels).float().unsqueeze(1) 


if __name__ == "__main__":

	import time

	# Numpy Dataset
	start = time.time()
	dataset = NumpyDataset(data_folder = './', 
							root_img=None, gender = True)
	end = time.time()
	print('%.4f s' % (end-start))

	# Get labels unique set
	print(dataset.get_labels_set())