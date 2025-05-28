Tips to fit scSpecies
=====================

1) We recommend training on an NVIDIA GPU or using Apple Metal Performance Shaders (MPS).

2) To reduce VRAM usage, consider using a negative binomial model instead of a zero-inflated negative binomial model. Additionally, modeling the dispersion parameter across batches or datasets rather than per cell can substantially lower memory demands.

3) To assess alignment quality in the latent space via alignment metrics it is necessary to match annotation of homologous cell types accross species. 

4) The length of the homologous gene sequence between the context and target datasets should either include more than 500 genes or contain prominent marker genes.

5) Unbounded activation functions, such as ReLU or PReLU, should be clipped to a finite interval (e.g., [-6, 6]) to ensure numerical stability. This can be achieved via the `layer_order` argument. Since scVI operates directly on raw gene expression count data, the large dynamic range across cells can otherwise lead to instability during model training. Additionally, we recommend using layer normalization and dropout in scVI network architectures.

6) When analyzing gene expression across three or more species, reinitialize and retrain the context decoder to prevent artifacts introduced by the softmax function, which now must normalize over varying sets of homologous genes.

7) Before fine-tuning, inspect the unaligned context latent space to ensure that cell clusters are well-separated.

8) Select the context dataset to be as comprehensive and large as possible. If the target dataset may include novel or underrepresented cell types, consider augmenting the context data with cells from a reference cell atlas.

9) For smaller datasets (< 10000 cells), scVI can overfit. We therefore recommend training for fewer epochs. 

10) When gene sequences are found to have no overlap by the package `mygene`, match gene symbols manually, e.g. by mapping names to the mouse or human genome. 