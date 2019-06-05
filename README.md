Implementation of implicit generative models based on the MMD in PyTorch (for Python 3.6+).
Very much a work in progress; feel free to get in touch if you want to use it.

Currently contains:

* [`igms.featurize`](igms/featurize.py): extract features of images from pretrained classifiers
  (`torchvision` or the pretrained L2-robust ones from [`locuslab/smoothing`](https://github.com/locuslab/smoothing/)).
* [`igms.kernels`](igms/kernels.py): various standard kernels, potentially on top of those features, with some
  fancy machinery to cache various sums and so on.
* [`igms.mmd`](igms/mmd.py) to estimate the MMD, run permutation tests based on the MMD,
  and unbiasedly estimate the variance of the MMD,
  from the `igms.kernels.LazyKernelPair` class.
* [`train_gmmn.py`](https://github.com/dougalsutherland/igms/blob/master/train_gmmn.py):
  train a Generative Moment Matching Networks
  ([Y. Li+ ICML-15](https://arxiv.org/abs/1502.02761) / [Dziugaite+ UAI-15](https://arxiv.org/abs/1505.03906)),
  i.e. just minimize the MMD with a fixed kernel between the model and the target distribution,
  but using much richer kernels than those papers used.
  
Soon:
* Generative Feature Matching Networks ([Santos+ 2019](https://arxiv.org/abs/1904.02762)),
  which are basically GMMNs with a moving-average trick to estimate the MMD.
* FID and [KID](https://arxiv.org/abs/1801.01401) evaluation (KID is essentially done, just need a wrapper).
* MMD three-sample tests like [Bounliphone+ ICLR-16](https://arxiv.org/abs/1511.04581) (essentially done).
* Optimized kernel two-sample tests like [Sutherland+ ICLR-17](https://arxiv.org/abs/1611.04488)
  &ndash; not really an IGM, but it's basically implemented already.
* Adaptive IGM learning rates based on the KID three-sample test,
  like in [Bińkowski+ ICLR-18](https://arxiv.org/abs/1801.01401).
  
Other things I might implement here eventually:

* (S)MMD-GANs:
  [C.-L. Li+ NIPS-17](https://arxiv.org/abs/1705.08584) 
  / [Bińkowski+ ICLR-18](https://arxiv.org/abs/1801.01401) 
  / [Arbel+ NeurIPS-18](https://arxiv.org/abs/1805.11565),
  which use a GAN-like adversarial game to choose the best kernel.
* Method of Learned Moments ([Ravuri+ ICML-18](https://arxiv.org/abs/1806.11006)),
  which is like a GMMN that (a) uses gradient features and
  (b) occasionally updates them based on retraining the classifier.
* Generative Latent Optimization ([Bojanowski+ 2017](https://arxiv.org/abs/1707.05776))?
