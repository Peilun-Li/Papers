# Papers
<img src="legend.png" height="30">

## Table of Contents

<!-- toc -->

- [CNN](#cnn)
  * [RCNN](#rcnn)
- [RNN](#rnn)
- [GAN](#gan)
  * [GAN](#gan-1)
- [other NN](#other-nn)
- [other ML](#other-ml)

<!-- tocstop -->

## CNN
### [RCNN](CNN/RCNN.pdf)
```
Ross Girshick, CVPR 2014
Workflow: Selective search for region proposals -> AlexNet top layer features -> SVM classification
Thoughts: Transfer learning from ImageNet, visualization, bias on positive samples
Concepts: Feature map, receptive field
To read: 
  P. Sermanet, K. Kavukcuoglu, S. Chintala, and Y. LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVPR, 2013. (unsupervised pretraining)
  J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. IJCV, 2013. (selective search)
  M. Zeiler, G. Taylor, and R. Fergus. Adaptive deconvolutional networks for mid and high level feature learning. In CVPR, 2011. (visualization)
  P. Felzenszwalb, R. Girshick, D. McAllester, and D. Ra- manan. Object detection with discriminatively trained part based models. TPAMI, 2010. (hard negative mining)
  K. Sung and T. Poggio. Example-based learning for view-based human face detection. Technical Report A.I. Memo No. 1521, Massachussets Institute of Technology, 1994. (hard negative mining)
```
## RNN

## GAN
### [GAN](GAN/GAN.pdf)
```
Ian J. Goodfellow, NIPS 2014
Workflow: 
  Training generator and discriminator alternatively using sgd and bp (with different step)
  Both G and D will be improved through training
  Goal: G can generate indistinguishable images compared with true images
Thoughts:
  A minimax optimization problem
  Sidestep difficulties in traditional CNN (by adding a D as opt. target rather than dataset)
  Min to max to avoid saturation 
  Reserve images distribution
Concepts: adversarial nets, generative model, discriminative model, Helvetica scenario (G maps too many noise input to the same output)
To read:
  Jarrett,K.,Kavukcuoglu,K.,Ranzato,M.,andLeCun,Y.(2009).Whatisthebestmulti-stagearchitecture for object recognition? In Proc. International Conference on Computer Vision (ICCV’09), pages 2146–2153. IEEE. (piecewise linear unit)
  Glorot, X., Bordes, A., and Bengio, Y. (2011). Deep sparse rectifier neural networks. In AISTATS’2011. (piecewise linear unit)
  Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013a). Maxout networks. In ICML’2013. (piecewise linear unit)
  Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2012b). Improving neural networks by preventing co-adaptation of feature detectors. Technical report, arXiv:1207.0580. (dropout)
  Breuleux, O., Bengio, Y., and Vincent, P. (2011). Quickly generating representative samples from an RBM-derived process. Neural Computation, 23(8), 2053–2073. (Gaussian Parzen window parameter estimation)
```
## other NN

## other ML
