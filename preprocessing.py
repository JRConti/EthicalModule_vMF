"""
Preprocessing on face embeddings.
Allows for adding a class/label attribute 
(such as gender) to data by filtering data. 

Jean-Rémy Conti
2022
"""

import os
import datetime
from pathlib import Path
import shutil
import numpy as np
from sklearn import preprocessing as skprepro
from tqdm import tqdm


class Preprocessing():
	'''
	'''
	def __init__(self, data_folder, working_dir, supp_attributes = None):
		'''

		Parameters
		----------
		data_folder: str
			Path to folder containing face embeddings/labels npy files,
			as well as img_list txt file.
		working_dir: str
			Working directory, location of preprocessed data.
		supp_attributes: str or list (of str)
			Names of attributes that have been previously extracted
			and synced with this class, that one wants to load
			without any checking of synchronization.
		'''

		# Check paths are ok
		assert Path(data_folder).is_dir(), 'The data folder does not exist.'
		if not Path(working_dir).is_dir():
			Path(working_dir).mkdir(parents= True, exist_ok= True)

		# First copy data from data_folder to working directory
		if not Path(working_dir + 'embeddings.npy').is_file():
			print('Copying files ...')
			shutil.copy2(data_folder + 'embeddings.npy', working_dir)
			shutil.copy2(data_folder + 'labels.npy', working_dir)
			shutil.copy2(data_folder + 'img_list.txt', working_dir)
			print('Files copied to working directory.\n')
		else:
			print('Files already in working directory. Continue ? [y/n]')
			ans = input()
			assert ans == 'y'
			print()

		self.original_folder = data_folder
		self.working_dir = working_dir 

		# Load labels (use of memmap for memory efficiency)
		self.data = dict()
		self.data['labels'] = np.load(self.working_dir + 'labels.npy', mmap_mode = 'r+') # -> type: np.memmap
		self.format_labels() # Re-arrange labels to be range(max_labels +1)

		# Load face embeddings
		self.data['embeddings'] = np.load(self.working_dir + 'embeddings.npy', mmap_mode = 'r+') # -> type: np.memmap
		assert self.data['embeddings'].shape[0] == self.n_data, 'Input data and corresponding labels have to share 1 dimension'

		with open(self.working_dir + 'img_list.txt', 'r') as f:
			self.img_list = f.readlines() 
		assert len(self.img_list) == self.n_data, 'Input data and corresponding image names should have same length'

		# Add previously extracted attribute data
		if supp_attributes is not None:
			if type(supp_attributes) != list:
				supp_attributes = [supp_attributes]

			for attribute_name in supp_attributes:
				attribute_path = self.working_dir + attribute_name + '.npy'
				assert Path(attribute_path).is_file(), 'There is no npy file located at: {}'.format(attribute_path)
				self.data[attribute_name] = np.load(attribute_path, mmap_mode = 'r+')
				assert self.data[attribute_name].shape[0] == self.n_data, 'Attribute {} data and corresponding labels have to share 1 dimension'.format(attribute_name)


	def replace_file(self, name, temp_name):
		'''
		Delete file located at self.working_dir + name and
		replace it by the file at self.working_dir + temp_name.
		'''

		path = self.working_dir + name
		temp_path = self.working_dir + temp_name

		os.remove(path)
		os.rename(temp_path, path)

	
	def change_npy(self, data_name, new_array):
		'''
		Update a npy file with new_array data.

		Parameters
		----------
		data_name: str
			Name of data concerned in npy modification.
			Must be in self.data.keys().
		new_array: np array
			At the end of this method, we have:
			self.data[data_name] = new_array
			(with memory map loading).
		'''

		# Save in temp file
		temp_name = data_name + '_temp'
		np.save(self.working_dir + temp_name, new_array)

		# Free memory map to update old data
		self.data[data_name] = None
		self.replace_file(data_name + '.npy', temp_name + '.npy')
		
		# Update data
		self.data[data_name] = np.load(self.working_dir + data_name + '.npy', mmap_mode = 'r+')


	def format_labels(self):
		'''
		Arrange labels so that the values are all the integers between 0 and max_labels.
		'''

		le = skprepro.LabelEncoder()
		le.fit(self.data['labels'])
		new_labels = le.transform(self.data['labels'])
		
		# Update data
		self.change_npy('labels', new_labels)

		# Maximum value of labels 
		self.n_labels = np.amax(self.data['labels']) + 1
		self.n_data = self.data['labels'].shape[0]
		assert set(self.data['labels']) == set(range(self.n_labels))
	
		print('Labels formatted.')


	def reduce_npy_files(self, indices_to_del):
		'''
		Reduce all npy data contained in self.data, by deleting
		a part of data indices. Data previously saved in npy files
		is also updated. 

		Parameter
		---------
		indices_to_del: list
		'''

		print('Reducing npy data ...')
		pbar = tqdm(total= len(list(self.data.keys())))
		for data_name in self.data.keys():
			
			# Reduce data
			new_array = np.delete(self.data[data_name], indices_to_del, axis= 0)

			# Update data
			self.change_npy(data_name, new_array)

			pbar.update(1)
		pbar.close()


	def reduce_txt_files(self, lines_to_del):
		'''
		
		Parameter
		---------
		lines_to_del: list
			Indices of lines to delete.
			Numbered from 0 -> max_line_txt_file -1.
		'''

		print('Reducing txt data ...')

		lines_to_del = set(lines_to_del)

		# Create new txt data
		temp_name = 'img_list_temp.txt'
		with open(self.working_dir + temp_name, mode = 'w') as f_w:
			pbar = tqdm(total= len(self.img_list))
			for idx, line in enumerate(self.img_list):
				if idx not in lines_to_del: # fast because queries within a set (// lists)
					f_w.write(line) # Keep \n
				pbar.update(1)
			pbar.close()

		# Update txt data
		self.replace_file('img_list.txt', temp_name)
		with open(self.working_dir + 'img_list.txt', 'r') as f:
			self.img_list = f.readlines() 


	def del_indices_data(self, indices_to_del):
		'''
		Delete some entries of data, so that all files in working
		directory are sync.

		Parameter
		---------
		indices_to_del: list
			Indices of data to delete.
		'''

		if type(indices_to_del) != list:
			indices_to_del = [indices_to_del]
		
		print('Deleting {} entries of data ...'.format(len(indices_to_del)))

		# Update npy files
		self.reduce_npy_files(indices_to_del)

		# Update TXT files
		self.reduce_txt_files(indices_to_del)

		# Re-arrange labels
		self.format_labels() 


	def del_labels(self, labels_list):
		'''
		Delete entries with given labels.

		Parameter
		---------
		labels_list: list (of ints)
			Labels to exclude of dataset.
		'''

		if type(labels_list) != list:
			labels_list = [labels_list]

		# Find corresponding indices of data
		print('Finding indices to delete {} labels ...'.format(len(labels_list)))

		bad_indices = []
		pbar = tqdm(total= len(labels_list))
		for label in labels_list:
			bad_indices.extend( list(np.arange(self.n_data)[self.data['labels'] == label]) ) 
			pbar.update(1)
		pbar.close()

		# Delete corresponding data
		self.del_indices_data(bad_indices) 
		

	def add_class_attribute(self, path_to_class_attribute, attribute_name, threshold=1.):
		'''
		Import a TXT file listing an attribute for each label/class in plain text.
		Adapt original and new data to align them and save a npy file for the 
		new attribute in working directory. 

		Binary attribute for now.

		Parameter
		---------
		path_to_class_attribute: str
			Path to TXT file with at least 4 columns, delimited with ',':
			0: Name of label in plain text
			2: Number of positive votes: number of images within a class
				whose attribute is 1.
			3: Number of negative votes: number of images within a class
				whose attribute is 0.
			1: Attribute of the class, decided using majority vote.  
		attribute_name: str
			Name of new key in dict self.data, listing npy files.
		threshold: float in [0,1]
			Among all labels, we keep only those for which the number n_min
			of minority votes is less than a fraction of the number 
			n_max of the majority votes : n_min <= threshold * n_max

		Output
		------
		self.working_dir + attribute_name + '.npy': npy file
			Contains array of attribute, aligned with embeddings and labels.
		'''

		# Get access to genders of text labels
		path_to_class_attribute = Path(path_to_class_attribute)
		assert path_to_class_attribute.is_file()

		# Find text labels associated to all labels in npy file (create dict)
		print('Building dict for text labels ...')

		# Save text labels for each label
		self.txt2label = dict()
		pbar = tqdm(total= self.n_data)
		for idx, label in enumerate(self.data['labels']):
			label_txt = self.img_list [idx].split('/')[0] 
			assert label_txt not in self.txt2label or (label_txt in self.txt2label and self.txt2label[label_txt] == label)
			if label_txt not in self.txt2label.keys():
				self.txt2label[label_txt] = label
			pbar.update(1)
		pbar.close() 
		assert len(set(self.data['labels'])) == len(self.txt2label) 

		# Initialize class attributes with same shape than self.embeddings, self.labels
		attribute = -1 * np.ones(self.n_data, dtype= int)

		# Find text labels in attribute txt file that are compatible with original npy file
		# Build attribute array, aligned with labels
		print('Building {} data ...'.format(attribute_name))

		labelstxt2attribute = dict()
		with path_to_class_attribute.open(mode='r') as f_r:
			pbar = tqdm(total = len(self.txt2label))
			# Read attribute txt file
			for i, line in enumerate(f_r):			
				line = line.split(' ')
				# Get info
				name, stats = line[0], line[1].split(',')
				current_attribute = float(stats[0])
				pos_votes = float(stats[1])
				neg_votes = float(stats[2])

				# Check that labels are valid and exist in current dataset 
				assert current_attribute in [0, 0.5, 1], '{}'.format(current_attribute)
				assert name in self.txt2label, '{}'.format(name)

				# Build attribute array 
				if float(min(pos_votes, neg_votes)) / float(max(pos_votes, neg_votes)) <= threshold:
					labelstxt2attribute[name] = current_attribute
					attribute[ self.data['labels'] == self.txt2label[name] ] = current_attribute
				pbar.update(1)
			pbar.close()

		# Save file in working directory using str attribute_name 
		np.save(self.working_dir + attribute_name, attribute)

		# Load new attribute with memmap
		self.data[attribute_name] = np.load(self.working_dir + attribute_name + '.npy', mmap_mode= 'r+')
		assert self.data[attribute_name].shape[0] == self.n_data

		# Now, all labels not present in attribute txt file (or not valid) must be deleted
		print('Finding indices of data with labels not valid with {} txt file ...'.format(attribute_name))
		# Get ordered list of labels (with repetition)
		bad_indices = []
		pbar = tqdm(total= len(self.img_list))
		for idx, _ in enumerate(self.img_list):
			label_txt = self.img_list [idx].split('/')[0] 
			if label_txt not in labelstxt2attribute:
				bad_indices.append(idx)
			pbar.update(1)
		pbar.close()

		# Deleting corresponding indices
		assert set( self.data[attribute_name] [bad_indices] ) == set([-1]), 'Big problem here.' 
		self.del_indices_data(bad_indices) 
		assert self.data[attribute_name] [ self.data[attribute_name] == -1 ].shape[0] == 0, 'Big problem here.'

		# Check that there is only 1 value of attribute within each class
		print('Verifying that {} are the same within 1 label ...'.format(attribute_name))
		pbar = tqdm(total= int(self.n_labels))
		for label in range(self.n_labels):
			att_vals = self.data[attribute_name][ self.data['labels'] == label ]
			att_val = att_vals[0]
			assert set(att_vals) == set([att_val])
			pbar.update(1)
		pbar.close()

		# Keep threshold in memory
		self.threshold = {}
		self.threshold[ attribute_name ] = threshold

		print('Attribute \'{}\' has been added to data.'.format(attribute_name))


	def save(self):
		'''
		Save a config file which details composition of dataset.
		'''

		to_write = []
		to_write.append(str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M")) + '\n')
		to_write.append(' '.join(list(self.data.keys())) + '\n\n')		
		to_write.append('Input data: ' + str(Path(self.original_folder).absolute()) + '\n\n')

		supp_attributes = [ attribute for attribute in list(self.data.keys()) if attribute not in ['embeddings', 'labels'] ]
		for attribute_name in supp_attributes:
			if self.threshold[attribute_name] is not None:
				to_write.append(attribute_name + ' threshold: ' + str(self.threshold[attribute_name]) + '\n\n')
		
		to_write.append('n_data: ' + str(self.n_data) + '\n')
		to_write.append('n_labels: ' + str(self.n_labels) + '\n\n')

		for attribute_name in supp_attributes:
			line = []
			line.append('\n' + attribute_name + ':\tProp. IDs\tProp. imgs\n')
			for att_val in list(set( self.data[attribute_name] )):
				att_val_n_labels = len(list(set( self.data['labels'][ self.data[attribute_name] == att_val ] )))
				att_val_labels_frac = 100.0 * float(att_val_n_labels) / float(self.n_labels)
				att_val_pop = np.ones(self.n_data)[ self.data[attribute_name] == att_val ].shape[0]
				att_val_pop_frac = 100.0 * float(att_val_pop) / float(self.n_data) 
				line.append(str(att_val) + '\t\t' + '%.2f' % att_val_labels_frac + ' (' + str(att_val_n_labels) + ')\t' + '%.2f' % att_val_pop_frac + ' (' + str(att_val_pop) + ')\n')
			to_write.append(''.join(line) + '\n')

		with open(self.working_dir + 'config_data.txt', mode = 'w') as f_w:
			for line in to_write:
				f_w.write(line)

		print('\nConfig file for data has been saved at: %s' % self.working_dir + 'config_data.txt')


if __name__ == "__main__":

	threshold = float(0.33)

	# Original path 	
	root = '../data/Arcface_MS1MV3_R100_fp16/'
	
	data_folder = root 
	working_dir = root + 'insight_face/gender_threshold_' + str(threshold).replace('.','_') + '/'
	
	supp_attributes = None #'genders' otherwise
	prepro = Preprocessing(data_folder, working_dir, supp_attributes= supp_attributes)
	
	# Add genders
	print('\n---- Add genders ----')
	path_to_genders = root + 'insight_face/genders_id.txt'
	prepro.add_class_attribute(path_to_class_attribute= path_to_genders, attribute_name= 'genders', threshold=threshold)
	
	# Do not comment
	prepro.save()
