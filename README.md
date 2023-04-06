# Ethical Module - vMF

Code for the paper "Mitigating Gender Bias in Face Recognition Using the von Mises-Fisher Mixture Model", accepted at ICML 2022.
https://proceedings.mlr.press/v162/conti22a/conti22a.pdf (or https://arxiv.org/abs/2210.13664 for the arXiv version).


<p align="center">
  <img src="https://github.com/JRConti/EthicalModule_vMF/blob/main/images/vMF_sphere.png">
</p>


<ins>Short abstract of the paper:</ins>

Motivated by geometric considerations, we mitigate gender bias in Face Recognition through a new post-processing methodology which transforms the deep embeddings of a pre-trained model to give more representation power to discriminated subgroups. It consists in training a shallow neural network, called the Ethical Module, by minimizing a Fair von Mises-Fisher loss whose hyperparameters account for the intra-class variance of each gender. Extensive numerical experiments on a variety of datasets show that the proposed method is state-of-the-art for the problem of gender bias mitigation in Face Recognition. 


The files `prepare_dataset.py` and `preprocessing.py` are used before the Ethical Module training. Briefly, from a training set of face images (labelled with identity), a gender predictor and a pre-trained Face Recognition backbone, it consists in saving the deep embeddings of each image by the pre-trained model, their identity label and their predicted gender. The Ethical Module can then be trained on those data (`train.py`).

As precised in `prepare_dataset.py`, face image datasets can be found at https://github.com/deepinsight/insightface/wiki/Dataset-Zoo. Pre-trained models can be found at https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch (along a gender predictor). Other pre-trained models can be found at https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode.

<ins>Contact:</ins>
jean-remy.conti@telecom-paris.fr
