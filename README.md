# Training Data Influences Modularity in Neural Networks

## Overview

Alex Haskel <sup>1</sup>, Daoyuan Qian <sup>2</sup> and Ila Fiete <sup>2</sup>
<sup>1</sup> Department of Computer Science, CUNY Hunter College, NY
<sup>2</sup> McGovern Institute for Brain Research, Massachusetts Institute of Technology,
<sup>2</sup> K. Lisa Yang Integrative Computational Neuroscience Center in the Yang-Tan Collective, Massachusetts Institute of Technology, 
<sup>2</sup>Department of Brain and Cognitive Sciences, Massachusetts Institute of Technology

Modular tasks can be decomposed into parts, with each sub-problem being dependent on separate properties of the input data, which can vary independently of each other. Neural networks can solve such tasks by internally mixing information about various properties of the input. By contrast, the brain contains circuits that exhibit modular structure and address such tasks by processing information through relatively independent channels. Such modular organization may support learning efficiency and robustness, while reducing catastrophic forgetting.
Previous work introduced the weighted-activity regularizer, which multiplicatively combines the weights and activations of artificial neurons, penalizing overactive strongly connected units and encouraging modular organization in networks faced with modular tasks. We studied how the selection of training samples affects modularity in networks equipped with the WA regularizer.
We trained networks to solve a numerical modular task and constructed training datasets exhibiting different relationships among input dimensions, including random baselines, strongly correlated datasets, datasets optimized to reduce statistical dependence among input properties, and datasets containing repeated contrasts, where each input dimension varies while others remain fixed. We measured the modularity of networks trained under each condition using spectral measures of how strongly network connectivity separated into modules.
Our results revealed that highly correlated datasets are ineffective at inducing modularity, but purposefully reducing statistical dependence does not guarantee it. Datasets containing repeated contrasts among input properties strongly outperformed random baselines at inducing modularity, and did so at relatively small sample sizes. Such datasets remained effective with the random removal of large numbers of samples, but this effect was largely reduced with the addition of a few random points breaking the previously established structure of contrasts.
We found that information-theoretic measures of statistical dependence can track aspects of what makes repeated contrasts among input properties effective, but not the exact structure that makes such training sets successful. Further work would involve mathematically characterizing the structural features of such modularity-inducing datasets more precisely, and directly exploiting them for dataset generation without the need for exhaustive combinatorial grid-like datasets. 

## Repository structure

The [modularisation_via_noise](./modularisation_via_noise) folder contains basic scripts for training a deep non-linear network under various regularisers and noises, and assessing modular connectivity structure using the random walk Laplacian matrix. The python notebook 'run_nonlinear.ipynb' allows producing data with different characteristics, runs the training with various settings, and saves many different metrics in the [results_nonlinear](./modularisation_via_noise/results_nonlinear) folder.

For explanatory panels of the main results, please see the [Summary_panels](./Summary_panels) directory, and the poster associated to this work in [Poster](./Poster). The notebook 'various_graphs.ipynb' in [results_nonlinear](./modularisation_via_noise/results_nonlinear) allows reproducing the graphs that were used for the explanatory panels and poster.

Due to the many experiments that I ran, these are included as a zip file release in []. If you wish to each of the 44 original experiments, with all their original data summaries, and detailed metrics, please download this file.

Please also see Daoyuan's original repo, which first implemented the weighted-activity regularizer, and which contains further discussion about modular neural networks that this work builds on:

https://github.com/dq219/NoiseMod


Qian, D., Liang, Q., & Fiete, I. (2026). Modular Connectivity in Neural Networks Emerges from Poisson Noise-Motivated Regularization and Promotes Robustness and Compositional Generalization. PRX Life, 4, 033022.
https://journals.aps.org/prxlife/abstract/10.1103/8bps-t2vq


## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
