# Papers
<img src="legend.png" height="30">

## Table of Contents

<!-- toc -->

- [CNN](#cnn)
  * [RCNN](#rcnn)
  * [SPP-net](#spp-net)
  * [Fast R-CNN](#fast-r-cnn)
  * [Faster R-CNN](#faster-r-cnn)
  * [Mask R-CNN](#mask-r-cnn)
  * [Mask-X-RCNN](#mask-x-rcnn)
  * [ImageNet CNN](#imagenet-cnn)
  * [Batch Normalization](#batch-normalization)
  * [Instance Normalization](#instance-normalization)
  * [Group Normalization](#group-normalization)
  * [Deconv Network](#deconv-network)
  * [CRN](#crn)
  * [Inception](#inception)
  * [Inception v2 & v3](#inception-v2--v3)
  * [Inception v4](#inception-v4)
  * [Xception](#xception)
  * [ResNet](#resnet)
  * [ResNeXt](#resnext)
  * [DenseNet](#densenet)
  * [ShuffleNet](#shufflenet)
  * [Survey of Semantic Segmentation](#survey-of-semantic-segmentation)
  * [FCN-SemanticSeg](#fcn-semanticseg)
  * [Multi Task Domain Adaptation](#multi-task-domain-adaptation)
  * [U-Net](#u-net)
  * [BagNets](#bagnets)
- [RNN](#rnn)
  * [RNN LM](#rnn-lm)
  * [BRNN](#brnn)
  * [deep RNN](#deep-rnn)
  * [Sequence to Sequence](#sequence-to-sequence)
  * [Efficient Estimation (word2vec)](#efficient-estimation-word2vec)
  * [Distributed Representations (word2vec)](#distributed-representations-word2vec)
- [GAN](#gan)
  * [GAN](#gan-1)
  * [Improved GAN](#improved-gan)
  * [DCGAN](#dcgan)
  * [SimGAN](#simgan)
  * [pix2pix](#pix2pix)
  * [CycleGAN](#cyclegan)
  * [Wasserstein GAN](#wasserstein-gan)
  * [CoGAN](#cogan)
  * [StarGAN](#stargan)
  * [Perceptual loss](#perceptual-loss)
  * [Learning not to learn](#learning-not-to-learn)
- [GNN](#gnn)
  * [GNN](#gnn-1)
- [other NN](#other-nn)
  * [Neural Style](#neural-style)
- [OCR (Document Understanding)](#ocr-document-understanding)
  * [Full page text recognition](#full-page-text-recognition)
- [other ML](#other-ml)
  * [GTAV](#gtav)
  * [Playing for Data](#playing-for-data)
  * [Playing for Benchmarks](#playing-for-benchmarks)
  * [HOG](#hog)
  * [SIFT](#sift)
  * [DNDF](#dndf)
  * [Selective Search](#selective-search)
  * [LM evaluation](#lm-evaluation)
  * [SceneNet RGB-D](#scenenet-rgb-d)
  * [SoDeep](#sodeep)
- [Slides](#slides)
  * [Word Vec CS224d-L2](#word-vec-cs224d-l2)
  * [Word Vec more CS224d-L3](#word-vec-more-cs224d-l3)
  * [MT with RNN CS224d-L9](#mt-with-rnn-cs224d-l9)
  * [Deep LSTM CS224d-L10](#deep-lstm-cs224d-l10)
  * [Advanced Recursive NN CS224d-L11](#advanced-recursive-nn-cs224d-l11)
  * [CNN for NLP CS224d-L13](#cnn-for-nlp-cs224d-l13)
  * [NN in SR CS224d-L14](#nn-in-sr-cs224d-l14)
  * [NMT CS224d-L15](#nmt-cs224d-l15)
  * [DMN CS224d-L17](#dmn-cs224d-l17)
  * [MDP CS234-L1](#mdp-cs234-l1)
  * [MDP to RL CS234-L2](#mdp-to-rl-cs234-l2)
  * [Monte Carlo and Generalization CS234-L3](#monte-carlo-and-generalization-cs234-l3)
  * [Model Free Methods and Approximation](#model-free-methods-and-approximation)
  * [RNN](#rnn-1)

<!-- tocstop -->

## CNN
### [RCNN](CNN/RCNN.pdf)
```
Rich feature hierarchies for accurate object detection and semantic segmentation
Ross Girshick, Berkeley, CVPR 2014
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
Kaiming He, Microsoft, TPAMI 2015
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
Ross Girshick, Microsoft,ICCV 2015
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
  K.He, X.Zhang, S.Ren, and J.Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV,2014. (SPPnets)
  S. Lazebnik, C. Schmid, and J. Ponce. Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories. In CVPR, 2006. (spatial paramid pooling)
SecretKey: Experiments results verify concerns, loss function matters, end-to-end often outperforms doing separately
```
### [Faster R-CNN](CNN/FasterRCNN.pdf)
```
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Microsoft, NIPS 2015
Workflow:
  Region Proposal Network
  Approximate joint training / Alternating training
Thoughts:
  Feature maps may be reused
Concepts: anchor
```
### [Mask R-CNN](CNN/MaskRCNN.pdf)
```
Mask R-CNN
Kaiming He, FAIR, ICCV 2017
Workflow:
  Extends Faster-RCNN: add a mask branch besides the box branch
  RoIAlign: no quantization by using bilinear interpolation 
Thoughts:
  Decouple mask and classification: one mask per class (binary loss on mask instead of multinomial loss)
  Artificial: 1000 RPN and only mask top 100
```
### [Mask-X-RCNN](CNN/Mask-X-RCNN.pdf)
```
Learning to Segment Every Thing
Ronghang Hu, BAIR & FAIR, CVPR 2018
Workflow:
  Partially supervised training paradigm with weight transfer function
  Based on Mask RCNN: transfer learning from box weights to mask weights
  Weight transfer function: class-agnostic
Thoughts:
  Instance segmentation with broad classes within the visual world
SecretKey: Speed
```
### [ImageNet CNN](ImageNet-CNN.pdf)
```
ImageNet Classification with Deep Convolutional Neural Networks
Alex Krizhevsky, UToronto, NIPS 2012
Workflow:
  ReLU, Multi GPU, Local Response Normalization
Thoughts:
  ReLU trains faster than tanh and sigmoid
```
### [Batch Normalization](CNN/BatchNorm.pdf)
```
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
Sergey Ioffe, Google, ICML 2015
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
### [Instance Normalization](CNN/InstanceNorm.pdf)
```
Workflow:
  Replace batch norm with instance norm (channel wise) to improve generative performance
  Applied on both train and test phases
Thoughts:
  Style transfer results should be independent of the contrast of input image
Concepts: instance normalization (a.k.a. contrast normalization)
To read:
  Ulyanov, D., Lebedev, V., Vedaldi, A., and Lempitsky, V. S. (2016). Texture networks: Feed-forward synthesis of textures and stylize images. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, pages 1349–1357. (style transfer)
```
### [Group Normalization](CNN/GroupNorm.pdf)
```
Group Normalization
Yuxin Wu, FAIR, ECCV 2018
Workflow:
  Group Norm: Group-level norm across channel dimension
    Better performance than batch norm when batch size is small 
Thoughts:
  Reuse "old" knowledge: group-wise normalization
  Group norm removes batch size constraint in batch norm
Concepts: group-wise normalization
To read:
  SIFT and HOG
  Y. LeCun, L. Bottou, G. B. Orr, and K.-R. Mu ̈ller. Efficient backprop. In Neural Networks: Tricks of the Trade. 1998. (normalizing makes training faster; initializing)
```
### [Deconv Network](CNN/Deconv.pdf)
See also "A giude to convolution arithmetic for deep learning" [here](CNN/ConvGuide.pdf)
```
Deconvolutional Networks
Matthew D. Zeiler, NYU, CVPR 2010
Workflow:
  Top-down: reverse of conv
```
### [CRN](CNN/CRN.pdf)
```
Photographic Image Synthesis with Cascaded Refinement Networks
Qifeng Chen, Stanford, ICCV 2017
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
### [Inception](CNN/Inception.pdf)
```
Going Deeper with Convolutions
Christian Szegedy, Google, CVPR 2015
Workflow:
  Inception models:
    Parallel & Concat filters: 1x1 conv, 1x1 & 3x3 conv, 1x1 & 5x5 conv, 3x3 maxpool & 1x1 conv
    1x1 conv before 3x3 and 5x5 conv to reduce computation
Thoughts:
  Sparse computation is much slowser than dense computation for today's computing (overhead of lookups and cache misses)
  How to approximate sparse structure using dense components -> parallel different conv sizes
  Middle layer features are discriminative -> add auxiliary classifiers on intermediate layers
  Auxiliary classifiers can combat gradient vanishing and provide regularization
  Training models on different crop sizes
Concepts: Hebbian principle (biology), Inception, GoogLeNet
To read:
  M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013. (1x1 conv, average pooling)
  A. G. Howard. Some improvements on deep con- volutional neural network based image classification. CoRR, abs/1312.5402, 2013. (model improvement)
SecretKey: dense<->sparse
```
### [Inception v2 & v3](CNN/Inception-v2v3.pdf)
```
Rethinking the Inception Architecture for Computer Vision
Christian Szegedy, Google, CVPR 2016
Workflow:
  Inception-v2:
    Replace 5x5, 7x7 conv with multi-layer 3x3 conv (holding same input & output shape)
    LSR: encourage model to be less confident
  Inception-v3:
    Add batch normalization for FC layer in auxiliary classifier, based on v2
Thoughts:
  Architectural improvements in NN can improve performance
  General NN design principles:
    Representation size should decrease from input to the output
    Increasing activations allows for more disentangled features
    Spatial aggregation in lower dimension embeddings won't result in much loss in representation power
    Balance width and depth of network (increase them in parallel can have optimal improvement)
  Auxiliary classifiers acts as regularizer
  Label smoothing regularization (hard label -> soft label)
  Gradient clipping is useful to stabilize training
Concepts: LSR, Inception-v2, Inception-v3
To read:
  K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.(VGGNet)
  T. Tieleman and G. Hinton. Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4, 2012. Accessed: 2015- 11-05. (RMSProp)
  R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. arXiv preprint arXiv:1211.5063, 2012. (gradient clipping)
SecretKey: encourge to be less confident (hard decision -> soft decision)
```
### [Inception v4](CNN/Inception-v4.pdf)
```
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
Christian Szegedy, Google, AAAI 2017
Workflow:
  Inception-v4: deeper and optimized version of inception v3
  Inception-ResNet-v2: combine inception with resnet
Concepts: Inception-v4, Inception-ResNet-v2
```
### [Xception](CNN/Xception.pdf)
```
Francois Chollet, Google, CVPR 2017
Workflow:
  Replace INception modules with depthwise separable convs with residual connections
  Absense of non-linearity within separable convs leads to faster training and performance
Thought:
  Extreme form of Inception module is almost identical to depthwise separable conv
```
### [ResNet](CNN/ResNet.pdf)
```
Deep Residual Learning for Image Recognition
Kaiming He, Microsoft, CVPR 2016
Workflow:
  Shorcut connections
Thoughts:
  Identity map via shortcut (add)
Concepts: deep residual learning
```
### [ResNeXt](CNN/ResNeXt.pdf)
```
Aggregated Residual Transformations for Deep Neural Networks
Saining Xie, Ross Girshick, UCSD & FB, CVPR 2017
Workflow:
  Resnet with Inception's split-transform-merge strategy within each shorcut connection
Thoughts:
  Inception: split-transform-merge strategy
  Increase cardinality (size of transformations set) may have better effect than increase depth and width
  Group convolution helps reduce number of parameters (allowing higher cardinality)
To read:
  A. Krizhevsky, I. Sutskever, and G. Hinton. Im- agenet classification with deep convolutional neural networks. In NIPS, 2012. (Group Convolution)
  M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR, 2014. (Network in network)
```
### [DenseNet](CNN/DenseNet.pdf)
```
Densely Connected Convolutional Networks
Gao Huang, Zhuang Liu, Cornell, CVPR 2017
Workflow:
  DenseNet: connect each layer to every other layer in a dense block
    Layers can be narrow (small number of filters per layer), forming a collective knowledge
    DenseNet-B: with bottleneck layer (reduce number of feature maps before 3x3 conv)
    DenseNet-C: with compression (reduce number of output feature maps in transition layer)
    DenseNet-BC: combine B and C, most parameter efficient
Thoughts:
  DenseNet can:
    Alleviate vanishing-gradient problem: easy to train
    Strengthen feature propagation
    Encourage fature reuse
    Reduce number of parameters: no need to relearn redundant feature maps
  Dense connections have a regularization effect (reduce overfitting)
  Feature reuse
  1x1 conv can be seen as changing number of filters
  Deep supervision: layers recive additional supervision from loss function through shorter connections
  Visualization on weights to validate shorter connections are useful
  DenseNet integrates identity mapping, deep supervision and diversified depth
Concepts: dense blocks, transition layers, growth rate, bottleneck layer, deep supervision
To read:
  Q. Liao and T. Poggio. Bridging the gaps between residual learning, recurrent neural networks and visual cortex. arXiv preprint arXiv:1604.03640, 2016. (resnet and rnn)
  C.-Y.Lee,S.Xie,P.Gallagher,Z.Zhang,andZ.Tu.Deeply- supervised nets. In AISTATS, 2015. (deep supervision)
  G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger. Deep networks with stochastic depth. (Stochastic depth regularization for resnet)
SecretKey: ResNet<->RNN
```
### [ShuffleNet](CNN/ShuffleNet.pdf)
```
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
Xiangyu Zhang, Xinyu Zhou, Face++, CVPR 2018
Workflow:
  Pointwise group convolution to reduce computation
  Channel shuffle to fully relate channels in different groups
  Use depthwise convolution
```
### [Survey of Semantic Segmentation](CNN/Survey_SemanticSeg.pdf)
```
A Survey of Semantic Segmentation
Martin Thoma
Workflow:
  SS is done typically with a classifier sliding on fixed-size feature inputs
To read:
  N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in Computer Vision and Pattern Recognition, 2005 (HOG)
  L. Bourdev, S. Maji, T. Brox, and J. Malik, “Detecting people using mutually consistent poselet activations,” in Computer Vision–ECCV 2010. (HOG)
  P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, “Object detection with discriminatively trained part-based models,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 32, no. 9, pp. 1627–1645, 2010. (HOG)
  D. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91–110, 2004. (SIFT)
  N. Plath, M. Toussaint, and S. Nakajima, “Multi- class image segmentation using conditional random fields and global classification,” in Proceedings of the 26th Annual International Conference on Machine Learning. ACM, 2009, pp. 817–824. (SIFT)
  G. Csurka, C. Dance, L. Fan, J. Willamowski, and C. Bray, “Visual categorization with bags of keypoints,” in Workshop on statistical learning in computer vision, ECCV, vol. 1, no. 1-22. Prague, 2004, pp. 1–2. (BOV)
  G. Csurka and F. Perronnin, “A simple high performance approach to semantic segmentation.” in BMVC, 2008, pp. 1–10. (BOV+SIFT)
  S.-C. Zhu, C.-E. Guo, Y. Wang, and Z. Xu, “What are textons?” International Journal of Computer Vision, vol. 62, no. 1-2, pp. 121–143, 2005. (texton)
  P. H. Pinheiro and R. Collobert, “Recurrent convolutional neural networks for scene parsing,” arXiv preprint arXiv:1306.2795, 2013. (recurrent CNN for SS)
SecretKey: link b/w ML and DL
```
### [FCN-SemanticSeg](CNN/FCN-SemanticSeg.pdf)
```
Fully Convolutional Networks for Semantic Segmentation
Jonathan Long, Evan Shelhamer, Berkeley, CVPR 2015
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
### [Multi Task Domain Adaptation](CNN/MultitaskDomAdapt.pdf)
```
Workflow:
  Multi-task domain adaptation for fine-grained recoginition
    Multi-task loss including: 
      Softmax loss on fine-grained and attribute level
      Attribute consistency loss
      Unsup/Semisup adaptation loss
    CaffeNet+ImageNet transfer
Thoughts:
  Convert class level loss to attribute level loss
  Symmetric version of KL divergence to min distance b/w dist.
To read:
  S.Ben-David, J.Blitzer, K.Crammer, F.Pereira, etal. Analysis of representations for domain adaptation. Advances in neural information processing systems, 19:137, 2007. (theoretical framework for domain adaptation)
  A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. (CNN architecture)
  G. A. Miller. Wordnet: a lexical database for english. Com- munications of the ACM, 38(11):39–41, 1995. (WordNet)
```
### [U-Net](CNN/U-Net.pdf)
```
U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger
Workflow:
  U-Net:
    Contracting path to capture context
    Symmetric expanding path for precise localization
  Mirror input image for border prediction
Concepts: overlap-tile strategy
```
### [BagNets](CNN/BagNets.pdf)
```
Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet
Wieland Brendel and Matthias Bethge, ICLR 2019
Workflow:
  Class activations on modified RestNet per image patch
  Evaluate all patches to get one heatmap per class (Similar as sliding window on feature representations)
  Average heat map and softmax to get class prediction
  Similar results as state-of-the-art DNNs, but slower
Thoughts:
  Improvement of DNNs is because of fine-tuning, not better strategies. (But still based on ResNet)
```
## RNN
### [RNN LM](RNN/RNNLM.pdf)
```
Recurrent neural network based language model
Tomas Mikolov, Johns Hopkins, Interspeech 2010
Workflow:
  Vanilla RNN
    Initial state set to vector of small values, e.g., 0.1
    Decrease learning rate if log-likelihood has no significant improvement
  Dynamic RNN
    Continue training during test phase
  All rare words are treated equally, i.e., uniform dist. between rare words
Thoughts:
  Initialization is not crucial for large amount of data
Concepts: dynamic model
```
### [BRNN](RNN/BRNN.pdf)
```
Bidirectional Recurrent Neural Networks
Mike Schuster and Kuldip K. Paliwal, IEEE, IEEE TRANSACTIONS ON SIGNAL PROCESSING, 1997
Workflow:
  RNN with positive time direction and negative time direction
    Forward states not connected to backward states (two hidden nodes)
To read:
  A. Waibel, T. Hanazawa, G. Hinton, K. Shikano, and K. J. Lang, “Phoneme recognition using time-delay neural networks,” IEEE Trans. Acoust., Speech, Signal Processing, vol. 37, pp. 328–339, Mar. 1989. (Time delay neural network)
```
### [deep RNN](RNN/deepRNN.pdf)
```
Speech Recognition with Deep Recurrent Neural Networks
Alex Graves (+Hinton), UToronto
Workflow:
  Deep RNNs: stacking hidden layer
  Training: Connectionist Temporal Classification (CTC), RNN Transducer
Thoughts:
  Training methods can make some problem possible
Concepts: deep RNN, deep LSTM, CTC
To read:
  A. Graves and J. Schmidhuber, “Framewise Phoneme Classification with Bidirectional LSTM and Other Neu- ral Network Architectures,” Neural Networks, vol. 18, no. 5-6, pp. 602–610, June/July 2005. (bidirectional LSTM)
  A. Graves, S. Ferna ́ndez, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Net- works,” in ICML, Pittsburgh, USA, 2006. (CTC)
  A. Graves, Supervised sequence labelling with recurrent neural networks, vol. 385, Springer, 2012. (CTC)
  A. Graves, “Sequence transduction with recurrent neu- ral networks,” in ICML Representation Learning Work- sop, 2012. (RNN Transducer)
```
### [Sequence to Sequence](RNN/seq2seq.pdf)
```
Sequence to Sequence Learning with Neural Networks
Ilya Sutskever, Google, NIPS 2014
Workflow:
  Tow deep LSTM: 
    input sequence -> vector with fixed dimension
    vector -> decode output sequence
  Deep LSTM outperforms shallow LSTM
  Reverse order of input sequence
  Decode: left-to-right beam search decoder
Thoughts:
  DNN can only handle fixed dimension -> LSTM map var length seq to fixed vector
  Reverse input seq order to introduce short term dependencies that makes optimization simpler
  LSTM tends to no suffer from gradient vanishment, but can have gradient explosion: enforce hard constraint on gradient norm
Concepts: BLEU score, beam-search decoder
To read:
  S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997. (LSTM)
  D. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986. (RNN)
  T. Mikolov, M. Karafia ́t, L. Burget, J. Cernocky`, and S. Khudanpur. Recurrent neural network based language model. In INTERSPEECH, pages 1045–1048, 2010. (RNN)
  M. Sundermeyer, R. Schluter, and H. Ney. LSTM neural networks for language modeling. In INTERSPEECH, 2010. (RNN)
  A. Graves. Generating sequences with recurrent neural networks. In Arxiv preprint arXiv:1308.0850, 2013. (Attention, LSTM formulation)
  D.Bahdanau, K.Cho, and Y.Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014. (Attention)
  P. Werbos. Backpropagation through time: what it does and how to do it. Proceedings of IEEE, 1990. (RNN)
  D. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986. (RNN)
  S. Hochreiter and J. Schmidhuber. LSTM can solve hard long time lag problems. 1997. (Minimal time lag)
SecretKey: TWO lstm(GAN?), var length sequence
Pending: input & output details
```
### [Efficient Estimation (word2vec)](RNN/efficient_est_word2vec.pdf)
```
Distributed Representations ofWords and Phrases and their Compositionality
Tomas Mikolov, Google, NIPS 2013
Workflow:
  CBOW (continuous BOW): predicting the word given its context
  Skip-gram: predicting the context given a word
```
### [Distributed Representations (word2vec)](RNN/distributed_rep_word2vec.pdf)
```
Distributed Representations of Words and Phrases and their Compositionality
Tomas Mikolov, Google, NIPS 2013
Workflow:
  Hierarchical softmax: O(n) to O(logn)
  Noise Contrastive Estimation (NCE) -> Negative sampling (NEG): max log prob. of softmax
  Subsampling of frequent words: frequent words have less information value than rare words
```
## GAN
### [GAN](GAN/GAN.pdf)
```
Generative Adversarial Nets
Ian J. Goodfellow, Montreal, NIPS 2014
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
  Jarrett, K., Kavukcuoglu, K., Ranzato, M., and LeCun, Y. (2009). What is the best multi-stage architecture for object recognition? In Proc. International Conference on Computer Vision (ICCV’09), pages 2146–2153. IEEE. (piecewise linear unit)
  Glorot, X., Bordes, A., and Bengio, Y. (2011). Deep sparse rectifier neural networks. In AISTATS’2011. (piecewise linear unit)
  Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013a). Maxout networks. In ICML’2013. (piecewise linear unit)
  Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2012b). Improving neural networks by preventing co-adaptation of feature detectors. Technical report, arXiv:1207.0580. (dropout)
  Breuleux, O., Bengio, Y., and Vincent, P. (2011). Quickly generating representative samples from an RBM-derived process. Neural Computation, 23(8), 2053–2073. (Gaussian Parzen window parameter estimation)
```
### [Improved GAN](GAN/improved-GAN.pdf)
```
Improved Techniques for Training GANs
Tim Salimans, openai, NIPS 2016
Workflow:
  Feature matching: matching G on activations of an intermediate layer of D
  Minibatch discrimination: An extra matrix (tensor) that produce extra side information of other examples in the minibatch by matrix (tensor) multiplication
  Historical averaging
  One-sided label smoothing
  Vitual batch normalization: normalize based on a reference batch
Thoughts:
  Game theory: training GANs require finding a Nash equilibrium of a non-convex game
  When collapse (Helvetica) is about to happen, D's gradient may point to similar directions for many similar points
Concepts: Nash equilibrium, virtual batch normalization
To read:
  AlecRadford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. (DCGAN)
  C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception Architecture for Computer Vision. ArXiv e-prints, December 2015. (label smoothing)
SecretKey: GAN from game theory
```
### [DCGAN](GAN/DCGAN.pdf)
```
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Alec Radford & Luke Metz, indico Research, ICLR 2016
Workflow:
  Model architectures that enable stable training, higher resolution and deeper generative models
    All convolutional net (replace spatial pooling with strided convolutions)
    Eliminated fully connected layers on top of convolutional features    
    Batch normalization
Thoughts:
  Global average pooling increases model stability but hurt convergence speed
  Batchnorm to all layers but not G's output layer and D's input layer to prevent sample oscillation and model instability
  Architecture guideline for stable DCGAN
    Replace pooling layers with strided conv (D) and fractional-strided conv (G)
    Use batchnorm in both G and D
    Remove fully connected hidden layers
    Use ReLU in G for all layers except output (Tanh)
    Use LeakyReLU in D for all layers
Concepts:
To read:
  Gregor, Karol, Danihelka, Ivo, Graves, Alex, and Wierstra, Daan. Draw: A recurrent neural network for image generation. arXiv preprint arXiv:1502.04623, 2015. (RNN for generativ models)
  Dosovitskiy, Alexey, Springenberg, Jost Tobias, and Brox, Thomas. Learning to generate chairs with convolutional neural networks. arXiv preprint arXiv:1411.5928, 2014. (deconvolution for generative models)
  Springenberg, Jost Tobias, Dosovitskiy, Alexey, Brox, Thomas, and Riedmiller, Martin. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014. (all convolutional net)
  Mordvintsev, Alexander, Olah, Christopher, and Tyka, Mike. Inceptionism : Going deeper into neural networks. (global average pooling)
  Mikolov, Tomas, Sutskever, Ilya, Chen, Kai, Corrado, Greg S, and Dean, Jeff. Distributed repre- sentations of words and phrases and their compositionality. (word2vec)
  Dosovitskiy, Alexey, Springenberg, Jost Tobias, and Brox, Thomas. Learning to generate chairs with convolutional neural networks. (conditional generative models)
SecretKey: reversed face embedding <-> visualization on GAN
```
### [SimGAN](GAN/SimGAN.pdf)
```
Learning from Simulated and Unsupervised Images through Adversarial Training
Ashish Shrivastava, Apple, CVPR 2017
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
Phillip Isola, Berkeley, CVPR 2017
Workflow:
  Generator: a encoder-decoder network with skip connections (U-Net)
  Conditional GAN (input image and random noise)
Thoughts:
  L1 encourages less blurring than L2
```
### [CycleGAN](GAN/CycleGAN.pdf)
```
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
Jun-Yan Zhu, Taesung Park, Berkeley, ICCV 2017 (submitted)
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
### [Wasserstein GAN](GAN/WGAN.pdf)
```
Wasserstein GAN
Martin Arjovsky
Workflow:
  Goal of unsup learning: min KL divergence b/w real data dist. and learned dist.
  EM distance is continuous and differentiable
  Optimized on EM distance, WGAN is more stable
Thoughts:
  Training GAN is delicate and unstable [To read 1]
Concepts: Earth Mover(EM) distance, Wasserstein-GAN
To read:
  Martin Arjovsky and L ́eon Bottou. Towards principled methods for training generative adversarial networks. In International Conference on Learning Rep- resentations, 2017. Under review.
```
### [CoGAN](GAN/CoGAN.pdf)
```
Workflow:
  Learn joint distribution of multi-domain images without paired training data
    By sharing weight: first layers of G, and last layers of D (A GAN for each domain)
Thoughts: Paired images in two domains share the same high-level concepts
```
### [StarGAN](GAN/StarGAN.pdf)
```
Workflow:
  Multidomain image-to-image translation: 
    Extra mask vector as input containing class label and dataset mask
    CycleGAN with an extra classification loss for multi-dataset multi-domain
  multi-task learning: only train a dataset's associated labels (not other datasets')
```
### [Perceptual loss](GAN/Perceptual-Loss.pdf)
```
Perceptual Losses for Real-Time Style Transfer and Super-Resolution
Justin Johnson, Stanford, EVVC 2016
Workflow:
  Use perceptual loss for style transfer: high level features from a pretrained network
    Feature reconstruction loss
    Style reconstruction loss
To read: 
  Vondrick, C., Khosla, A., Malisiewicz, T., Torralba, A.: Hoggles: Visualizing ob- ject detection features. In: Proceedings of the IEEE International Conference on Computer Vision. (2013) 1–8 (HOG loss)
```
### [Learning not to learn](GAN/Learning_Not_to_Learn.pdf)
```
Learning Not to Learn: Training Deep Neural Networks with Biased Data
Byungju Kim, CVPR 2019
Workflow:
  Regulate a network to minimize mutual information between extracted features and bias
    Mutual information -> reformulated as auxiliary dist. -> relaxed to KL divergency -> relaxed to cross-entropy loss
    Classification loss + (adversarial fashion) bias prediction loss + bias loss
Thoughts:
  Remove bias information will help learn useful features
```
## GNN
### [GNN](GNN/GNN.pdf)
```
Relational inductive biases, deep learning, and graph networks
Peter W. Battaglia, DeepMind
Workflow:
  Key design principles:
    Flexible representations: structure could be either set or inferred (or in between)
    Configurable within-block structure: computation steps and functions could be changed
    Composable multi-block architectures: GN blocks can be stacked
  Computation steps (order could be changed depending on tasks):
    Update edges; 
    Edges effect on nodes; 
    Update node; 
    Summing edges for global attributes; 
    Summing nodes for global attributes; 
    Update global attributes
  Implement:
    Nodes and edges updates could be treated like the batch dimension for mini-batch training
  Open questions:
    How to init graph structure?
    How to modify graph structure adaptively during computation?
Thoughts:
  Biology: nature join with nurture <-> AI: structure join with flexibility
  Representations and structure of entities and relations can be learnt instead of explicit definition
  Inductive biase often trades flexibility with complexity
  Graph supports arbitrary relational structure, providing better inductive bias than CNN/RNN
  Relational inductive biases in GN:
    Representations/Relations are defined in input, rather than the architecture (as in CNN/RNN)
    GN is invariant to permutations, and supports combinatorial generalization (infinite use of finite means)
Concepts: relational inductive biases, GN block (graph-to-graph module)
To read:
  Watters, N., Zoran, D., Weber, T., Battaglia, P., Pascanu, R., and Tacchetti, A. (2017). Visual interaction networks: Learning a physics simulator from video. In Advances in Neural Information Processing Systems, pages 4542–4550. (init graph structure)
  van Steenkiste, S., Chang, M., Greff, K., and Schmidhuber, J. (2018). Relational neural expectation maximization: Unsupervised discovery of objects and their interactions. Proceedings of the International Conference on Learning Representations (ICLR). (init graph structure)
  Li, Y., Vinyals, O., Dyer, C., Pascanu, R., and Battaglia, P. (2018). Learning deep generative models of graphs. In Workshops at the International Conference on Learning Representations (ICLR). (init graph structure, modify graph structure in runtime)
  Kipf, T., Fetaya, E., Wang, K.-C., Welling, M., and Zemel, R. (2018). Neural relational inference for interacting systems. In Proceedings of the International Conference on Machine Learning (ICML). (init graph structure, modify graph structure in runtime)
```
## other NN
### [Neural Style](other_NN/Neural-Style.pdf)
```
A Neural Algorithm of Artistic Style
Leon A. Gatys
Workflow:
  Learn a image that fits representations of style image and content image
    Square error loss on multilayer's feature maps (extra Gram matrix for style representation loss)
    Jointly minimize content loss and style loss, from a white image and pretrained VGG-19
Thoughts: min feature level loss
```
## OCR (Document Understanding)
### [Full page text recognition](OCR/Full-Page-Text-Recognition.pdf)
```
Full-Page Text Recognition: Learning Where to Start and When to Stop
Bastien Moysset
Workflow:
  Localization step: detect start of text lines
  Recognization step: recognize and find where to stop
Concepts: 2D-LSTM, Convolutional RNN
To read:
  Bluche, T.: Joint line segmentation and transcription for end-to-end handwritten paragraph recognition. In: Advances in Neural Information Processing System (2016) (Attention mechanism w/o localization)
  Graves, A., Schmidhuber, J.: Offline handwriting recognition with mul- tidimensional recurrent neural networks. In: NIPS (2009) (2D-LSTM)
  Moysset, B., Kermorvant, C., Wolf, C.: Learning to detect and localize many objects from few examples. arXiv preprint arXiv:1611.05664 (2016) (CRNN)
  Erhan, D., Szegedy, C., Toshev, A., Anguelov, D.: Scalable object detection using deep neural networks. In: IEEE Conf. on Computer Vision and Pattern Recognition (2014) (training)
```
## other ML
### [GTAV](other_ML/GTAV.pdf)
```
Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?
Matthew Johnson-Roberson, UMich, ICRA 2017
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
Stephan R. Richter, Vibhav Vineet, TU Darmstadt & Intel, ECCV 2016
Workflow:
  Collect image and segmentation data from GTA V
  Mixing training segmentation model with virtual data and real world data
Concepts: detouring
```
### [Playing for Benchmarks](other_ML/PlayingForBenchmarks.pdf)
```
Playing for Benchmarks
Stephan R. Richter
Workflow:
  250K high-resolution video frames from GTA V using real-time rendering pipelines
```
### [HOG](other_ML/HOG.pdf)
```
Histograms of Oriented Gradients for Human Detection
Navneet Dalal, CVPR 2005
Workflow:
  Histograms of oriented gradient
Thoughts:
  Local contrast normalization is essential for good performance
Concepts: Histograms of Oriented Gradient (HOG)
```
### [SIFT](other_ML/SIFT.pdf)
```
Distinctive Image Features from Scale-Invariant Keypoints
David G. Lowe, IJCV 2004
Workflow:
  Scale Invariant Feature Transform
    Scale-space extrema detection
    Keypoint localization
    Orientation assignment
    Keypoint descriptor
Concepts: SIFT
```
### [DNDF](other_ML/DeepNDF.pdf)
```
Deep Neural Decision Forests
Peter Kontschieder, Microsoft Research, ICCV 2015
Workflow:
  Decision tree with stochastic routing
  Replace fc layer with NDF
Thoughts: deep learning & traditional machine learning
```
### [Selective Search](other_ML/Selective-Search.pdf)
```
Selective Search for Object Recognition
J.R.R. Uijlings, IJCV 2013
Workflow:
  Combine:
    Exhaustive search: capture all possible locations
    Segmentation: use image structure information
  Diversification: color spaces, similarity measures, starting regions
```
### [LM evaluation](other_ML/LM-evaluation.pdf)
```
Evaluation Metrics for Language Models
Stanley Chen, Douglas Beeferman, Ronald Rosenfeld, CMU
Workflow:
  Perplexity of a language model: inverse of geometric avg on test data
  Word-error rate: e.g., on speech recognition
Thoughts:
  Expected word accuracy / Word error rate is (near) linear with LOG perplexity
SecretKey: log-linear
```
### [SceneNet RGB-D](other_ML/SceneNet-RGB-D.pdf)
```
SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth
John McCormac, ICCV 2017
Workflow:
  Drop objects from the ceiling -> automated camera trajectories -> rendering & groundtruth
```
### [SoDeep](other_ML/SoDeep.pdf)
```
SoDeep: a Sorting Deep net to learn ranking loss surrogates, Martin Engilberge, CVPR 2019
Workflow:
  Learn differentiable surragate functions
Thoughts:
  Surrogate loss for non-differentiable metric -> Train a network to learn that metric for differentiable loss
```
## Slides
### [Word Vec CS224d-L2](Slides/CS224d-Lecture2.pdf)
```
Workflow:
  Word to Vector:
    Window based cooccurence matrix as vectors + SVD
    Word2Vec: skip-gram model with max log prob opt.
      Each word has two vectors (outside and center vectors)
      Approx. with negative sampling
    GloVe: combine count based (cooccur. matrix) and direct prediction (word2vec)
Thoughts:
  Neighbors can represent words
To read: 
  GloVe: Global Vectors for Word Representation
```
### [Word Vec more CS224d-L3](Slides/CS224d-Lecture3.pdf)
```
Workflow:
  Word2Vec:
    Skip-gram: Approx w/ binary LR b/w true pairs and random pairs (dist.)
    CBOW: predict center word from sum of surrounding word vectors
    Word2Vec: sum of center vector and outside vector
  Evaluation:
    Intrinsic: e.g., argmax cos dist. -> semantic and syntactic analogy questions
    Extrinsic: e.g., named entity recognition
```
### [MT with RNN CS224d-L9](Slides/CS224d-Lecture9.pdf)
```
Workflow:
  Encoder outputs a final hidden vector
  Decoder's hidden state is computed by a function of:
    Previous hidden state
    Encoder's final output hidden vector
    Previous predicted output word
```
### [Deep LSTM CS224d-L10](Slides/CS224d-Lecture10.pdf)
```
Workflow:
  Deep LSTM
  Recursive NN vs. Recurrent NN
```
### [Advanced Recursive NN CS224d-L11](Slides/CS224d-Lecture11.pdf)
```
Workflow:
  Recursive NN for:
    Phrase detection
    Sentiment detection
  Tree LSTM
```
### [CNN for NLP CS224d-L13](Slides/CS224d-Lecture13.pdf)
```
Workflow:
  Each filter focuses on a different n-gram
    Input n word vectors (concatenated), output a vector
  Max-over-time pooling
    Max a (scalar) activation for each vector
  Dropout & Softmax
Thoughts: CNN is similar as Recursive NN
```
### [NN in SR CS224d-L14](Slides/CS224d-Lecture14.pdf)
```
Workflow:
  NN in speech recognition:
    HMM-DNN
    CTC objective function
```
### [NMT CS224d-L15](Slides/CS224d-Lecture15.pdf)
```
Workflow:
  Seq2seq: details on forward & backward
  Advancing NMT:
    Vocabulary size: copy mechanism
    Sentence length: attention mechanism
    Language complexity (multi-word, informal spelling, etc.): character-level translation
```
### [DMN CS224d-L17](Slides/CS224d-Lecture17.pdf)
```
Workflow:
  Obstacles:
    For NLP no single model architecture can have consistent good results across tasks
    Fully joint multitask learning is hard
Thoughts:
  All NLP/AI tasks can be readuced to question answering
Concepts: Dynamic Memory Networks (DMN)
To read:
  Ask Me Anything: Dynamic Memory Networks for Natural Language Processing (Kumar et al., 2015)
  Dynamic Memory Networks for Visual and Textual Question Answering (Xiong et al., 2016)
```
### [MDP CS234-L1](Slides/CS234-Lecture1.pdf)
```
Workflow:
  RL: learn to make good sequences of decisions
  RL involves:
    Optimization
    Generalization: policy maps past experience to action
    Exploration
    Delayed consequences
  Markov decision process: <States, Actions, Reward model, T dynamics model, Y discount factor>
```
### [MDP to RL CS234-L2](Slides/CS234-Lecture2.pdf)
```
Workflow:
  Q(s, a) values:
    Expected discounted sum of rewards over H step horizon
    If start with action a and follow policy pi
  MDP (Markov Decision Process):
    Value Iteration(VI): iteration on Values
    Policy Iteration(PI): iteration on Policy
      Policy evaluation: calc V for all s
      Policy improvement: find state-action Q value by following policy forever for each state s; take argmax of Q
    PI takes fewer iterations but more expensive per iteration, VI the opposite
  MDP vs. RL:
    MDP: given model of stochastic outcomes
    RL: learn model of outcomes
  RL:
    Model-based passive RL: estimate MDP model parameters from data (count & average)
    Model-free passive RL: only maintain estimate of Q
      TD (Temporal Difference) learning on V (running average)
    Q-Learning:
      Update Q(s, a) every time experience (s, a, s', r): running average on new sample estimate on s'
      Expore vs. Exploit: E-greedy (like pagerank random jump) with decay over time
```
### [Monte Carlo and Generalization CS234-L3](Slides/CS234-Lecture3.pdf)
```
Workflow:
  Monte Carlo policy evaluation: uses empirica mean returen instead of expected return
  Incremental Monte Carlo Updates
  Value function approximation (VFA): replace value lookup table with general parameterized form, to scale to large state spaces
  Action-value function approximation
```
### [Model Free Methods and Approximation](Slides/CS234-Lecture4.pdf)
```
  Feature selection:
    Domain knowledge
    Flexible set of features & regularize
```
### [RNN](Slides/cs231n_2019_lecture10.pdf)
```
Video: https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=11&t=0s
Output is derived from hidden state
Truncated backpropagation through time: minibatch through time
Attention: two inputs and outputs 
  Target, e.g., vocabulary distribution to sample, and, 
  Attention, e.g., location distributions to focus on in next time step)
Multilayer RNNs: usually not deep (3-4 layers)
LSTM
  Extra c hidden states besides h hidden state
  ifog: i(nput) gate, f(orget) gate, o(utput) gate, g(ate) gate
RNN vs LSTM: LSTM has an extra C hidden state, enabling uninterrupted gradient flow through the very beginning 
  Prevent gradient exploding: sigmoid and addition (LSTM or GRU), or gradient clipping in vanilla RNN 
  Prevent gradient vanishment: C state, init f gate to 1 (do not forget initially)
  C state behaves like skip connections in ResNet
```
