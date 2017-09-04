# Papers
<img src="legend.png" height="30">

## Table of Contents

<!-- toc -->

- [CNN](#cnn)
  * [RCNN](#rcnn)
  * [SPP-net](#spp-net)
  * [Fast R-CNN](#fast-r-cnn)
  * [Faster R-CNN](#faster-r-cnn)
  * [Batch Normalization](#batch-normalization)
  * [CRN](#crn)
  * [FCN-SemanticSeg](#fcn-semanticseg)
- [RNN](#rnn)
- [GAN](#gan)
  * [GAN](#gan-1)
  * [Improved GAN](#improved-gan)
  * [SimGAN](#simgan)
  * [pix2pix](#pix2pix)
  * [CycleGAN](#cyclegan)
- [other NN](#other-nn)
- [other ML](#other-ml)
  * [GTAV](#gtav)
  * [Playing for Data](#playing-for-data)

<!-- tocstop -->

## CNN
### [RCNN](CNN/RCNN.pdf)
```
Rich feature hierarchies for accurate object detection and semantic segmentation
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
### [SPP-net](CNN/SPPnets.pdf)
```
Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
Kaiming He, TPAMI 2015
Workflow:
  Generate fixed-length representation regardless of image size/scale
    SPP layer between C and FC to allow inputting different image sizes
Thoughts:
  Training/Running a detector on feature maps is a popular idea
Concepts: Spatial pyramid pooling
SecretKey: CNN requires fixed input size (RNN?)
```
### [Fast R-CNN](CNN/FastRCNN.pdf)
```
Fast R-CNN
Ross Girshick, ICCV 2015
Workflow:
  Network:
    Input image and set of object proposals
    RoI pooling layer to extract fixed length feature vector, based on conv feature map on input image, and proposal coords
    Feed feature vector into fc, with two sibling output layers: softmax class prob, and bounding box regression for each class
  Transfer learning from ImageNet pretrianed networks
  Normalize ground truth regression targets
  Multi-task loss and scale invariant bounding box regression offsets
Thoughts:
  Sharing computation can speed up
  RoI layer is a speciall case of spatial pyramid pooling layer in SPPnets
  Truncated SVD can speed up fc layers
  Multi-task training improves classification accuracy
  Deep ConvNets are adept at learning scale invariance directly
To read:
  K.He,X.Zhang,S.Ren,andJ.Sun.Spatialpyramidpooling in deep convolutional networks for visual recognition. In ECCV,2014. (SPPnets)
  S. Lazebnik, C. Schmid, and J. Ponce. Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories. In CVPR, 2006. (spatial paramid pooling)
SecretKey: Experiments results verify concerns, loss function matters, end-to-end often outperforms doing separately
```
### [Faster R-CNN](CNN/FasterRCNN.pdf)
```
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren
Workflow:
  Region Proposal Network
  Approximate joint training / Alternating training
Thoughts:
 Feature maps may be reused
Concepts: anchor
```
### [Batch Normalization](CNN/BatchNorm.pdf)
```
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
Sergey Ioffe, ICML 2015
Workflow:
  Normalize each scalar feature independently for each mini-batch, and
  Extra linear transformation which can represent identity transformation together with the normalization part
  Add before nonlinearity (ReLU, sigmoid, etc.)
  Use global mean and var of training data for inference
  BN network adapation:
    Increase learning rate
    Remove dropout
    Reduce L2 weight regularization
    Accelerate learning rate decay
    Remove local response normalization
    Shuffle training examples more thoroughly
    Reduce photometric distortions
Thoughts:
  Eliminate internal covariate shift can speed up training
  Ensemble
  Training converges faster if inputs are whitened
  BN enables higher learning rates:
    Resilient to parameter scale (prevent explosion or vanishment)
    Prevent getting stuck in the saturated regimes of nonlinearities
  BN regularizes the model:
    A training example will be influenced/regularized by other examples within a mini-batch
Concepts: internal covariate shift
To read:
  Sutskever, Ilya, Martens, James, Dahl, George E., and Hinton, Geoffrey E. On the importance of initialization and momentum in deep learning. In ICML (3), volume 28 of JMLR Proceedings, pp. 1139–1147. JMLR.org, 2013. (momentum)
  Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic optimization. J.Mach. Learn. Res., 12:2121–2159,July 2011. ISSN 1532-4435. (adagrad)
SecretKey: tranfer usage of normalization
```
### [CRN](CNN/CRN.pdf)
```
Photographic Image Synthesis with Cascaded Refinement Networks
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
### [FCN-SemanticSeg](CNN/FCN-SemanticSeg.pdf)
```
Fully Convolutional Networks for Semantic Segmentation
Jonathan Long, Evan Shelhamer, CVPR 2015
Workflow:
  Fully convolutional neural network for semantic segmentation:
    Transfer learning from ImageNet
    Convert classification networks to FCN (FC to C like sliding window)
    Pixel-wise final convolution to produce heapmap
    Deconvolution (bilinear upsample) on output heatmap to generate segmentation
    A skip architecture to generate detailed segmentation (deconvolution trainable here)
    First train converted FCN, then add skip architecture(s) to upgrade
Thoughts:
  Semantic segmentation is a combination of semantics (what) and location (where)
  Reinterpret techniques as equivalent network modifications
  Convert fully connected layer to convolutional layer (which produces heatmap)
  ILSVRC init of upper layers is important
Concepts: receptive field, deconvolution, line search
To read:
  J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. DeCAF: A deep convolutional acti- vation feature for generic visual recognition. In ICML, 2014. (transfer learning)
  M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014, pages 818–833. Springer, 2014. (transfer learning)
  A. Giusti, D. C. Cires ̧an, J. Masci, L. M. Gambardella, and J. Schmidhuber. Fast image scanning with deep max-pooling convolutional neural networks. In ICIP, 2013. (fast scanning)
SecretKey: complicated NN is still simpler than hand crafted rules (adding edges/links may be closer to brain); knowledge can be widely applied (SemanticSeg & Detection)
```
## RNN

## GAN
### [GAN](GAN/GAN.pdf)
```
Generative Adversarial Nets
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
### [Improved GAN](GAN/improved-GAN.pdf)
```
Workflow:
  Feature matching: matching G on activations of an intermediate layer of D
  Minibatch discrimination: An extra matrix (tensor) that produce extra side information of other examples in the minibatch by matrix (tensor) multiplication
  Historical averaging
  One-sided label smooting
  Vitual batch normalization: normalize based on a reference batch
Thoughts:
  Game theory: training GANs require finding a Nash equilibrium of a non-convex game
  When collapse (Helvetica) is about to happen, D's gradient may point to similar directions for many similar points
Concepts: Nash equilibrium, virtual batch normalization
To read:
  AlecRadford,LukeMetz,andSoumithChintala.Unsupervisedrepresentationlearningwithdeepconvolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. (DCGAN)
  C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception Architecture for Computer Vision. ArXiv e-prints, December 2015. (label smooting)
SecretKey: GAN from game theory
```
### [SimGAN](GAN/SimGAN.pdf)
```
Learning from Simulated and Unsupervised Images through Adversarial Training
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
### [pix2pix](GAN/pix2pix.pdf)
```
Image-to-Image Translation with Conditional Adversarial Networks
Phillip Isola, CVPR 2017
Workflow:
  Generator: a encoder-decoder network with skip connections (U-Net)
  Conditional GAN (input image and random noise)
Thoughts:
  L2 encourages less blurring than L2
```
### [CycleGAN](GAN/CycleGAN.pdf)
```
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
Jun-Yan Zhu, Taesung Park, ICCV 2017 (submitted)
Workflow:
  Two G and two D form a cycle
  A cycle consistency loss
  History of generated images to reduce model oscillation (moving average)
  Identity mapping for painting->photo, to preserve color composition
Thoughts:
  GAN mapping is highly under-constrained
  Can be viewed as training two autoencoders
Concepts: adversarial loss, cycle consistency loss
To read:
  Many (unpaired image to image translation)
SecretKey: autoencoders reuse, identity mapping (resnet)
```
## other NN

## other ML
### [GTAV](other_ML/GTAV.pdf)
```
Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?
Matthew Johnson-Roberson, ICRA 2017
Workflow:
  Collect simulated data and annotation from GTA V
  Transfer learning from ImageNet pretrained VGG16 using only simulated data
  Faster-RCNN
Thoughts:
  Simulated data can achieve high level performance without assistance of real data
Concepts: depth buffer, stencil buffer
```
### [Playing for Data](other_ML/PlayingForData.pdf)
```
Playing for Data: Ground Truth from Computer Games
Stephan R. Richter, Vibhav Vineet, ECCV 2016
Workflow:
  Collect image and segmentation data from GTA V
  Mixing training segmentation model with virtual data and real world data
Concepts: detouring
```
