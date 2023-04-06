"""
Script to compute vMF constants.

Jean-Remy Conti
2022
"""

import mpmath
import numpy as np
import torch

def vmf_logpartition(d, kappa):
	'''
	Evaluates the log-partition log C_d(kappa) for vMF density.
	Inspired from: https://github.com/minyoungkim21/vmf-lib
	
	Parameters
	----------
	d: scalar (> 0)
		Dimension in which the vMF density is computed.
	kappa: torch tensor (N,)
		Concentration parameter of the vMF density.

	Returns:
	--------
	logC: torch tensor (N,) 
		Log-partition of the vMF density : log C_d(kappa)
	'''

	besseli = np.vectorize(mpmath.besseli)
	log = np.vectorize(mpmath.log)

	# Fix mpmath precision
	mpmath.dps = 50
	
	with torch.no_grad():
		s = 0.5*d - 1
		mp_s = mpmath.mpf(1) * s
		mp_kappa = mpmath.mpf(1) * kappa.detach().cpu().numpy()

		# log I_s(kappa)
		mp_logI = log( besseli(mp_s, mp_kappa) )        
		# From mpmath to torch 
		logI = torch.from_numpy( np.array(mp_logI.tolist(), dtype= float))
		
		if (logI!=logI).sum().item() > 0:  # there is nan
			raise ValueError('NaN is detected from the output of log-besseli()')

		logC = -0.5 * d * np.log(2*np.pi) + s * kappa.log() - logI
		
	return logC, logI


if __name__ == "__main__":

	import matplotlib.pyplot as plt

	d = 512

	kappas = torch.arange(1,100)
	logpartitions, logI = vmf_logpartition(d, kappas) 

	fig, ax = plt.subplots(1,2, figsize=(12,5))

	# plot logI
	ax[0].plot(kappas.numpy(), logI.numpy())
	ax[0].set_xlabel(r'$\kappa$', fontsize=14)
	ax[0].set_title(r'$\log(I_{\frac{d}{2}-1}(\kappa))$', fontsize=15)
	# plt.savefig('logI.pdf')
	# plt.savefig('logI.png')
	
	# plot logC
	ax[1].plot(kappas.numpy(), logpartitions.numpy())
	ax[1].set_xlabel(r'$\kappa$', fontsize=14)
	ax[1].set_title(r'$\log(C_d(\kappa))$', fontsize=15)
	fig.tight_layout()
	# plt.savefig('logpartition.pdf')
	# plt.savefig('logpartition.png')
	
	# plot logits bounds
	plt.figure()
	ub = logpartitions.numpy() + kappas.numpy()
	lb = logpartitions.numpy() - kappas.numpy()
	plt.plot(kappas.numpy(), ub, label=r'$\log(C_d(\kappa)) + \kappa$', color='seagreen', linestyle= 'dashed', linewidth=2)
	plt.plot(kappas.numpy(), logpartitions.numpy(), label= r'$\log(C_d(\kappa))$', color='seagreen')#+ kappas.numpy()*(-0.05)
	plt.plot(kappas.numpy(), lb, label=r'$\log(C_d(\kappa)) - \kappa$', color='seagreen', linestyle= 'dotted', linewidth=2)
	plt.fill_between(kappas.numpy(), lb, ub, color='seagreen', alpha=0.2)
	plt.xlabel(r'$\kappa$', fontsize=14)
	plt.legend()
	# plt.savefig('logpartition_bounds.pdf')
	# plt.savefig('logpartition_bounds.png')

	plt.show()