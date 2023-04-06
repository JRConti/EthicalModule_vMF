"""
Module placed after a pre-trained model to remove its bias (e.g. gender
bias). A very simple network transforms embeddings of the pre-trained
model into new embeddings, more fair. The training of the network is 
achieved with the Fair von Mises-Fisher loss. 

ArcFace ResNet pre-trained models can be found at: 
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

MobileFaceNet pre-trained models can be found at: 
https://github.com/JDAI-CV/FaceX-Zoo/blob/main/training_mode/README.md

During training, use EthicalModule to debias. The complete model is
obtained using FairModel with the pretrained model and ethical_mod.model
as arguments.

For inference, use MLP (with saved weights) and the pre-trained model
as arguments of FairModel. 

Jean-Remy Conti
2022
"""

import torch
import torch.nn.functional as F

from vmf_loss import vMFLoss
from backbones import get_model


class MLP(torch.nn.Module):
	'''
	A simple network to transform face embeddings of a pre-trained model.
	'''
	def __init__(self, D_in, H, feature_dim):
		'''
		
		Parameters
		----------
		D_in: int
			Dimension of input vector
		H: int
			Dimension of hidden layer
		feature_dim: int
			Dimension of output feature vector
		'''
		super(MLP, self).__init__()

		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear_end = torch.nn.Linear(H, feature_dim)

	def forward(self, x):

		h_relu = self.linear1(x).clamp(min=0)
		# No activation function at the end of the computation of 
		# face embeddings
		f = self.linear_end(h_relu) 
		return f

class EthicalModule(torch.nn.Module):
	'''
	Module acting as a add-on for a pre-trained model, trained with 
	vMF loss, to produce more fair face embeddings.
	'''
	def __init__(self, model, num_classes, kappas = None, labels_set= None):
		'''
		
		Parameters
		----------
		model: instance of torch.nn.Module
			Network which transforms embeddings of pre-trained model
			into new embeddings. 
		Other parameters are specific for vMFLoss.
		'''
		super(EthicalModule, self).__init__()

		self.model = model

		# Get the output dimension of the last linear layer of model
		self.feature_dim = list(self.model.children())[-1].out_features

		# Initialize loss
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.vmf_loss = vMFLoss(self.feature_dim, num_classes, kappas, labels_set).to(device)

	def forward(self, x, forward = True, return_loss = True, labels = None, get_features=False):
		'''
		
		Parameters
		----------
		x: input shape (N, feature_dim)
		forward: boolean
			Set to True to compute a forward pass on self.model with 
			the input. If False, only the loss is computed.
		return_loss: boolean
			Set to True to get the vMF loss w.r.t. the labels given.
		labels: None or tensor shape (N,)
			Only used if return_loss is True.
			Labels used for supervised learning.
		get_features: boolean
			Set to True to get the normalized features from the model
			as an output but without autograd.
		'''
		assert forward or not get_features 

		if forward:
			x = self.model(x)

		assert x.shape[1] == self.feature_dim, 'Input has shape {} and feature dimension is {}'.format(x.shape, self.feature_dim)

		L, pred = self.vmf_loss(x, return_loss = return_loss, labels = labels)
		
		if not get_features:
			return L, pred
		else:
			with torch.no_grad():
				f_tilde = F.normalize(x, p=2, dim=1)
			return L, pred, f_tilde.detach()

	def get_centroids(self):
		return self.vmf_loss.get_centroids()


class FairModel(torch.nn.Module):
	'''
	A complete model (pre-trained + add-on to debias) taking images as
	input and which outputs debiased face embeddings.
	'''
	def __init__(self, model, add_on):
		'''
		Parameters
		----------
		model: instance of torch.nn.Module
			Model to be debiased.
		add_on: instance of torch.nn.Module
			Add_on to the pre-trained model to reduce its bias. During
			training of the add-on, use ethical_mod.model. During 
			inference, use instance of MLP with saved weights.
		'''
		super(FairModel, self).__init__()
		self.model = model
		self.add_on = add_on

	def forward(self, x):
		x = self.model(x)
		x = self.add_on(x)
		return x


def load_fair_model(model_path, network, eth_mod_path):
	'''
	Useful function to load ArcFace pre-trained model and the trained ethical
	module MLP and return the complete fair model for testing phase.
	To be adapted if pre-trained model and ethical module's 
	architectures are modified.

	Parameters
	----------
	model_path: str
		Path to ArcFace pre-trained model checkpoint.
	network: str
		Type of ResNet pre-trained network (r100, r50, r34, r18).
	eth_mod_path: str
		Path to ethical module checkpoint.
	
	Returns
	-------
	fair_model: instance of torch.nn.Module
	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# load ArcFace pre-trained model
	pre_trained_model = get_model(network, fp16=False)
	pre_trained_model.load_state_dict(torch.load(model_path, map_location= device))
	pre_trained_model.eval()

	# load ethical module
	eth_model = MLP(512, 1024, 512)
	eth_model.load_state_dict(torch.load(eth_mod_path, map_location= device))
	pre_trained_model.eval()

	# get fair model
	fair_model = FairModel(pre_trained_model, eth_model)
	fair_model.eval()
	return fair_model


def load_fair_model_conf(backbone_type, backbone_conf_file, model_path, eth_mod_path):
	'''
	Useful function to load MobileFaceNet pre-trained model from FaceX-Zoo configuration and the 
	trained ethical module MLP and return the complete fair model for testing phase.
	To be adapted if pre-trained model and ethical module's architectures are modified.

	Parameters
	----------
	backbone_type: str
		Resnet, MobileFaceNet ...
	backbone_conf_file: str
		The path of backbone_conf.yaml.
		(see https://github.com/JDAI-CV/FaceX-Zoo/blob/main/training_mode/README.md)
	model_path: str
		Path to pre-trained model checkpoint.
	eth_mod_path: str
		Path to ethical module checkpoint.
	
	Returns
	-------
	fair_model: instance of torch.nn.Module
	'''

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# load pre-trained model
	from load_model_FaceX_Zoo.model_loader import ModelLoader
	from load_model_FaceX_Zoo.backbone.backbone_def import BackboneFactory
	backbone_factory = BackboneFactory(backbone_type, backbone_conf_file)
	model_loader = ModelLoader(backbone_factory)
	pre_trained_model = model_loader.load_model(model_path)
	pre_trained_model.eval().to(device)

	# load ethical module
	eth_model = MLP(512, 1024, 512)
	eth_model.load_state_dict(torch.load(eth_mod_path, map_location= device))
	pre_trained_model.eval()

	# get fair model
	fair_model = FairModel(pre_trained_model, eth_model)
	fair_model.eval()
	return fair_model

if __name__ == "__main__":

	D_in = 3
	H  =3
	feature_dim = 2

	# module = MLP(D_in, H, feature_dim)
	# # Access list of submodules
	# print(list(module._modules.items()))
	# print()

	# model = EthicalModule(module, num_classes=4, kappas = 14)
	# print(model(torch.tensor([5.4,3.1, 2.2]).unsqueeze(0).float(), forward= True, return_loss= False, labels= None, get_features= True))
	# # torch.save(model, 'test.pt')
	# # model = torch.load('test.pt')
	# print(model)
	# print('allo', model.feature_dim)
	# # assert 1 == 0

	# print('linear_end' in model._modules)
	# #for i in range(len(model._modules)):

	# # Access list of submodules
	# print(list(model._modules.items()))
	# print(model._modules['model']._modules)
	# print('model' in model._modules)
	# print(hasattr(model,'_modules'))
	# print(hasattr(model._modules,'_modules'))
	
	# for name, param in model.named_parameters():
	# 	print(name)
	# 	print(param)
	# 	print()
	# 	if name == 'vmf_loss.fc.weight':
	# 		print('Centroids')
	# 	else:
	# 		print()

	# print(model.model)

	path_model = 'dummy_model.pt'
	checkpoint_p = '50.pt'

	pre_trained_model = get_model('r100', fp16=False)

	module = MLP(D_in=512, H=512, feature_dim=512)
	model = EthicalModule(module, num_classes=92596, kappas = 14)
	state_dict = torch.load(checkpoint_p, map_location='cpu')
	model.load_state_dict(state_dict)
	
	# model = torch.load(path_model, map_location= 'cpu') # load architecture
	# model.load_state_dict(torch.load(checkpoint_p, map_location= 'cpu'))  

	complete_model = FairModel(pre_trained_model, model)
	complete_model.eval()