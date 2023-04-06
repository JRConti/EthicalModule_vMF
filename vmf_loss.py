"""
Pytorch implementation of von Mises-Fisher loss (vMF loss).
Data are projected on the unit hypersphere, each class fills a portion
of the sphere. It is possible to control the spread of each class on
the unit hypersphere. 

Jean-Remy Conti
2022
"""

import torch
import torch.nn.functional as F

from utils.vmf_logpartition import vmf_logpartition


class vMFLoss(torch.nn.Module):

	def __init__(self, feature_dim, num_classes, kappas, labels_set= None):
		'''

		Parameters
		----------
		feature_dim: int
			Dimension of features to project on hypersphere.
		num_classes: int
			Number of different classes to learn.
		kappas: float or list of floats 
			float: Concentration parameter of vMF loss (unique value for
			all classes).
			list of floats: Concentration parameters of vMF loss.
			* if the length of the list == num_classes:
				The i-th element of the list is the concentration 
				parameter for the i-th label/class.
			* if the length of the list < num_classes:
				Each parameter is associated to a value taken by an 
				external attribute (given in labels_set thereafter). 
				The ordering of the input list must correspond to the
				ordering of attribute values: the i-th element of the 
				list kappas is the concentration parameter for labels
				having attribute value equal to the i-th element in the
				sorted list of attributes (given in labels_set 
				thereafter). Ex: each sample has a binary (0/1) attribute
				(provided in labels_set thereafter), kappas = [10, 20]
				-> labels for which the attribute is 0 have concentration
				parameter equal to 10, labels for which the attribute is
				1 have concentration parameter equal to 20.
		labels_set: None or tensor of shape (num_classes, 2)
			Only necessary if kappas is a list with more than 
			1 element and whose length < num_classes.
			tensor (num_classes, 2):
			* column 0: unique label values 0 -> num_classes-1.
			* column 1: corresponding attribute values.
		'''
		super(vMFLoss, self).__init__()

		# Unique concentration parameter
		if type(kappas) in [float, int]:
			self.kappas = torch.ones(num_classes, dtype= float) * kappas
			self.logpartitions = torch.zeros(num_classes, dtype= float)
		# Several concentration parameters
		else:
			assert len(kappas) <= num_classes, ('The length of the list ' + 
				'kappas should not be greater than {}'.format(num_classes))

			# Concentration parameters for each attribute value
			if len(kappas) < num_classes:
				assert labels_set is not None, ('labels_set must be provided'+ 
				' if concentration parameters are assigned depending on an ' + 
				'attribute.')
				assert len(labels_set[:,0]) == num_classes,( 
				'Column 0 of labels_set must have size {}'.format(num_classes)
				)
				assert len(torch.unique(labels_set[:,0])) == num_classes, (
				('Column 0 of labels_set must contain {} distinct ' + 
				'labels'.format(num_classes)))
				attribute_values = torch.unique(labels_set[:,1], sorted= True)
				assert len(kappas) == len(attribute_values), ('The number of '+
				'unique attribute values must equal the length of kappas.')

				self.kappas = torch.zeros(num_classes, dtype= float)
				for idx in range(len(kappas)):
					self.kappas[ 
					labels_set[:,1] == attribute_values[idx] ] = kappas[ idx ]
				assert torch.all(self.kappas != 0), (
					'{}'.format(set(self.kappas.numpy())))
			# Concentration parameters for each label
			else:
				self.kappas = torch.FloatTensor(kappas)

			# Computation of logpartitions
			kappa_values = torch.unique(self.kappas)
			logpartition_values = vmf_logpartition(feature_dim, kappa_values)

			self.logpartitions = torch.zeros(num_classes, dtype= float)
			for idx in range(len(kappa_values)):
				self.logpartitions[ 
				self.kappas == kappa_values[idx] ] = logpartition_values[ idx ]
			assert torch.all(self.logpartitions != 0)

		self.kappas.unsqueeze_(0).requires_grad_(False)
		self.logpartitions.unsqueeze_(0).requires_grad_(False)

		self.feature_dim = feature_dim
		self.num_classes = num_classes
		self.fc = torch.nn.Linear(feature_dim, num_classes, bias=False)
		torch.nn.init.xavier_normal_(self.fc.weight)
		self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

	def forward(self, x, return_loss = True, labels = None):
		'''

		Parameters
		----------
		x: tensor shape (N, feature_dim)
		return_loss: boolean
			Set to True to return the loss w.r.t. the labels given after.
		labels: None or tensor shape (N,)
			Only used if return_loss is True.
			Labels used for supervised learning.

		Returns
		-------
		tuple (L, pred)
		L: None if labels is None.
		0d tensor otherwise
			vMF loss corresponding to x and labels.
		pred: torch shape (N,)
			Predicted class/label for x.
		'''
		
		if return_loss:
			assert labels is not None, "Provide labels to compute loss."
			assert len(x) == len(labels), (
				'Inputs and labels must share 1 dimension.')
			# Check labels
			assert torch.min(labels) >= 0
			assert torch.max(labels) < self.num_classes
		else:
			assert labels is None, (
			"No need to provide labels if loss is not computed.")

		# Normalize feature vectors, shape (Nxd)
		N = x.shape[0]
		assert x.shape[1] == self.feature_dim
		x = F.normalize(x, p=2, dim=1)
		assert x.shape == (N,self.feature_dim)

		# Shape (NxC)
		cos = self.fc(x)
		assert cos.shape == (N, self.num_classes)

		# Get norm of centroids  
		mu_norm = torch.norm(self.fc.weight, p=2, dim=1).unsqueeze(0)   
		assert mu_norm.shape == (1,self.num_classes)		

		# Matrix of cosines between feature i and centroid j (NxC)
		cos = cos / mu_norm # broadcasting division in PyTorch 
		assert cos.shape == (N, self.num_classes)
		# Ensure that we have real cosines
		cos = cos.clamp(min=-1.0,max=1.0)

		# Matrix of log C_d(kappa_j) + kappa_j cos theta_ij (NxC)
		logits = self.logpartitions.to(cos.device) + cos * self.kappas.to(cos.device)
		assert logits.shape == (N, self.num_classes)

		# Vector of predicted labels (Nx1)
		pred = torch.max(logits, dim = 1)[1] 

		if return_loss:
			L = self.ce_loss(logits, labels)
		else:
			L = None

		return L, pred

	def get_centroids(self):
		'''
		Return
		------
		centroids: tensor shape (Cxd)
			Tensor that lists all centroids, so that they are ordered 
			by increasing corresponding label. Ex:
			| centroid label 0 |
			| centroid label 1 |
			|        .         |
			|        .         |
			|centroid label C-1|			
		'''
	
		#Normalize centroids
		with torch.no_grad():
			centroids = F.normalize(self.fc.weight, p=2, dim=1)
			assert centroids.shape == (self.num_classes, self.feature_dim)

		return centroids.detach()


if __name__ == '__main__':

	torch.manual_seed(42)

	feature_dim = 512
	num_classes = 1000

	kappas = [10, 20, 200, 400]
	
	labels_set = None
	n_att_val = 4
	labels_set = torch.stack([torch.arange(num_classes), torch.cat(int(num_classes/n_att_val)*[torch.arange(n_att_val)])], dim=1)
	print(labels_set)
	print(labels_set.size())
	vmf_loss = vMFLoss(feature_dim, num_classes, kappas, labels_set)

	print(vmf_loss.kappas)
	print(vmf_loss.logpartitions)
	print(vmf_loss.fc)
	for param in vmf_loss.fc.parameters():
		print(param.size())

	N = 1024
	x = torch.randn((N, feature_dim))
	labels = torch.randint(0, num_classes, (N,))
	print(vmf_loss(x, return_loss = True, labels= labels))