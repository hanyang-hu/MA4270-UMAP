# MA4270 Computational Assignment

## Project Summary

In this project, we first use concentrations of measures to understand the curse of dimensionality for high-dimensional Euclidean spaces, then we summarize the computational aspect of a non-linear dimension reduction technique - [UMAP](https://arxiv.org/abs/1802.03426), and test its [parametric version](https://arxiv.org/abs/2009.12981) on synthetic and real-world data sets (with a PyTorch-based implementation from scratch). We use insights from the tangent spaces of differentiable manifolds to effectively estimate the intrinsic dimension of the underlying manifold structure of the data and hence automatically determine the embedding dimension. 

**Remark.** This implementation was created for personal learning purposes only and is not optimized for practical applications. It is more recommended to use the [official implementation](https://github.com/lmcinnes/umap) of parametric UMAP (which should also support PyTorch). Please use it at your own discretion.