# EthicalModule-vMF

Code for the paper "Mitigating Gender Bias in Face Recognition Using the von Mises-Fisher Mixture Model", accepted at ICML 2022.
https://proceedings.mlr.press/v162/conti22a/conti22a.pdf (or https://arxiv.org/abs/2210.13664 for the arXiv version).


<p align="center">
  <img src="https://github.com/JRConti/EthicalModule_vMF/blob/main/images/vMF_sphere.png">
</p>


<ins>Short abstract of the paper:</ins>

Motivated by geometric considerations, we mitigate gender bias in Face Recognition through a new post-processing methodology which transforms the deep embeddings of a pre-trained model to give more representation power to discriminated subgroups. It consists in training a shallow neural network by minimizing a Fair von Mises-Fisher loss whose hyperparameters account for the intra-class variance of each gender. Extensive numerical experiments on a variety of datasets show that the proposed method is state-of-the-art for the problem of gender bias mitigation in Face Recognition. 
