"""
Get face embeddings along with gender predictions of a RecordIO dataset 
composed of: a rec file, a idx file and a lst file. Those 3 files must 
be saved at the same location with the same name (ex: train.rec, 
train.idx, train.lst).

Computes gender statistics for each identity and keep only identities
with enough confidence about the gender of all its images.

Gender predictions and regularization have been DISABLED !

Outputs
-------
* folder args.output:
	embeddings.npy
		Binary file listing all face embeddings of RecordIO dataset 
		with pre-trained model.
	labels.npy
		Binary file listing corresponding labels (identities).
	genders.npy
		Binary file listing corresponding genders, predicted by a 
		trained model.
	img_list.txt
		Names of images, ordered as previous npy files.
	genders_id.txt
		File which lists for each unique identity :
		- the number of images for which gender = 0
		- the number of images for which gender = 1
		- the majority vote decision
	genders_id_log.txt
		Log file for the computation of gender statistics per identity.

* folder args.output + 'threshold_' + args.threshold:
(same files as above, except that less data are present: only identities
with enough confidence about the gender of all its images)
	embeddings.npy
	labels.npy
	genders.npy
	img_list.txt
	config_data.txt
		Summary of data properties.

Jean-Rémy Conti
2022
"""

# Datasets can be found at: 
#	https://github.com/deepinsight/insightface/wiki/Dataset-Zoo
# Pre-trained models can be found at: 
# 	https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
# A gender predictor can be found in the Insight Face Github repo.

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# inference
from utils.image_iter_rec import FaceImageIter 
from backbones import get_model

# gender regularization
# from pre_trained.gender_stats import gender_statistics
# from preprocessing import Preprocessing

# Shuffle during training only via dataloader

batch_size = 256 		

parser = argparse.ArgumentParser(
	description='Get face embeddings of rec dataset with pre-trained model.')
parser.add_argument('-p', '--path', default=None, type=str, 
	help='Path to rec file')
parser.add_argument('-m', '--model', default='../model/softmax,50', 
	help='Path to load model.')
parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU ids', 
	nargs='+')
parser.add_argument('-n', '--network', type=str, default='r50', 
	help='Type of backbone network (r18, r34, r50, r100)')
parser.add_argument('-o', '--output', default=None, type=str, 
	help='Path to output folder')
parser.add_argument('-t', '--threshold', default=1., type=float, 
	help='Gender regularizer, float in [0,1]. Among all identities, we ' + 
	'keep only those for which the number n_min of minority gender votes ' + 
	'is less than a fraction of the number n_max of the majority gender ' + 
	'votes : n_min <= threshold * n_max ')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Output folder
Path(args.output).mkdir(parents= True, exist_ok= True)
print('Output folder:', args.output)

# Read dataset
data_iter = FaceImageIter(batch_size = batch_size, 
						data_shape = (3,112,112), 
						path_imgrec= args.path)
assert Path(args.path[0:-3] + 'lst').is_file(), (
	'RecordIO dataset must contain a list of images file at {}.'.format(
												args.path[0:-3] + 'lst')
												)
with open(args.path[0:-3] + 'lst', 'r') as f_r:
	n_img = len(f_r.readlines())
	assert len(data_iter.imgidx) == n_img, (
	'rec file contains {} images while lst file has {}.'.format(
												len(data_iter.imgidx), n_img)
											)

# Load pretrained model
model = get_model(args.network, fp16=False)
model.load_state_dict(torch.load(args.model, map_location= device))
model.eval().to(device)
if type(args.gpu) == list and len(args.gpu) > 1:
    # multi-GPU setting
    model = torch.nn.DataParallel(model, device_ids = args.gpu).to(device)

# Load gender predictor 
# ...


# Inference
print('\n-------------- Inference --------------\n')
embeddings = []
labels = []
# genders = []
pbar = tqdm(total=len(data_iter.imgidx))
for batch in data_iter:
	data = batch.data[0].asnumpy() # shape (b,c,h,w), RGB color code 
	# data_iter is now empty 
	if data.shape[0] == 0:
		break
	label = batch.label[0].asnumpy().astype(int)
	labels.extend(list(label))

	# # Gender predictions
	# data_np = np.moveaxis(data, 1, -1) # shape (b,h,w,c), RGB color code
	# for img in data_np:
	# 	gender = ...
	# 	genders.append(gender)

	# Inference
	with torch.no_grad():
		data = torch.from_numpy(data).to(device)
		
		# Preprocessing of pre-trained model 
		data.div_(255).sub_(0.5).div_(0.5) # values in [-1,1]
	
		# Inference
		output = model(data).cpu().numpy()
		embeddings.extend(output)
	pbar.update(batch_size)
pbar.close()

# Save embeddings, labels and genders
np.save(args.output + 'embeddings', embeddings)
np.save(args.output + 'labels', labels)
# np.save(args.output + 'genders', genders)

# Copy list of image file and get ground-truth labels
with open(args.path[0:-3] + 'lst', 'r') as f_r:
	img_lst = f_r.readlines()
labels = []
with open(args.output + 'img_list.txt', 'w') as f_w:
	for line in img_lst:
		label = int(line.split('\t')[-1])
		line = line.split('\t')[1].split('images/')[-1] + '\n'
		labels.append(label)
		f_w.write(line)

# Sanity check on img/embeddings order
labels_gt = np.load(args.output + 'labels.npy')
assert len(labels) == labels_gt.shape[0], '{} {}'.format(
											len(labels), labels_gt.shape[0]
														)
for i in range(len(labels)):
	assert labels[i] == labels_gt[i], '{} {}'.format(labels[i], labels_gt[i])


# # Gender regularization
# print('\n-------- Gender regularization --------\n')

# # Compute gender statistics per identity
# gender_statistics(genders_p= args.output + 'genders.npy', 
# 				list_p= args.output + 'img_list.txt')

# # Regularization
# working_dir = (args.output + 'gender_threshold_' + 
# 			str(args.threshold).replace('.','_') + '/')
# prepro = Preprocessing(data_folder= args.output, working_dir= working_dir)
# # Add genders
# print('\n---- Add genders ----')
# prepro.add_class_attribute(
# 	path_to_class_attribute= args.output + 'genders_id.txt', 
# 	attribute_name= 'genders', 
# 	threshold= args.threshold)
# # Do not comment
# prepro.save()