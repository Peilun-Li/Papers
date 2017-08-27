# Papers
<img src="legend.png" height="30">

## Table of Contents

<!-- toc -->

- [CNN](#cnn)
  * [RCNN](#rcnn)
  * [CRN](#crn)
- [RNN](#rnn)
- [GAN](#gan)
  * [GAN](#gan-1)
  * [SimGAN](#simgan)
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
### [CRN](CNN/CRN.pdf)
```
Qifeng Chen, ICCV 2017
Workflow: 
  An end-to-end CNN to generate synthetic images from semantic layouts
    Progress refinement: start from small images, multiple CNN modules to upsample
    Modified loss that encourages generating diverse images in one shot. (akin to k-means)
  Dataset: (photo, semantic segmentation)
Thoughts:
  An inverse of semantic segmentation (an underconstrained one-to-many problem)
  Coordinate global structure from low resolution (when distant objects are close)
  Progress refinement to generate high resolution
  Model capacity is essential for generating high resolution synthetic images
  Loss function between original image and synthetic image: COMPARE CNN LAYERS FEATURES (content representation)
  A loss akin to k-means, to encourage generating diverse images.
Concepts: Cascaded Refinement Network (CRN), reference image (original image)
To read:
  M. Arjovsky and L. Bottou. Towards principled methods for training generative adversarial networks. In ICLR, 2017. (GAN's unstable)
  L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein. Unrolled generative adversarial networks. In ICLR, 2017. (GAN's unstable)
  L.A.Gatys,A.S.Ecker,andM.Bethge.Imagestyletransfer using convolutional neural networks. In CVPR, 2016. (content representation)
  A. Nguyen, J. Yosinski, Y. Bengio, A. Dosovitskiy, and J. Clune. Plug & play generative networks: Conditional iter- ative generation of images in latent space. In CVPR, 2017. (conditional synthesis of diverse images)
SecretKey: content representation for annotation
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
### [SimGAN](GAN/SimGAN.pdf)
```
Ashish Shrivastava, CVPR 2017
Workflow: 
  A standard GAN with
    Self-regularization term (to preserve annotation)
    Local adversarial loss (to avoid artifacts) 
    Images history for training discriminator
  Fully CNN for both R and D for pixel level refining and local adversarial loss
Thoughts:
  Labeling large datasets is expensive and time consuming -> use refined synthetic images
  Self-regularization loss to preserve annotations (CAN BE CUSTOMIZED)
  Single strong discriminator may "overfit" -> use local combinations (local adversarial loss)
  "Moving average" on synthetic images for discriminator's training to stablize
  Similar receptive field for R and D (R uses ResNet)
Concepts: Simulated+Unsupervised(S+U) learning, refiner network
To read:
  A. Gaidon, Q. Wang, Y. Cabon, and E. Vig. Virtual worlds as proxy for multi-object tracking analysis. In Proc. CVPR, 2016. (pretraining on synthetic data)
  T. Salimans, I. Goodfellow, W. Zaremba, V. Che- ung, A. Radford, and X. Chen. Improved techniques for training gans. arXiv preprint arXiv:1606.03498, 2016. (running average on parameters for GAN training)
SecretKey: moving average, Img2Vec
```
## other NN

## other ML
