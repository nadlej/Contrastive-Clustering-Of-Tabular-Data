# Contrastive Clustering Of Tabular Data

## Overview
This repository comprises the codebase utilized in the preparation of my master's thesis. You can access the thesis via the link provided below. The comprehensive summary 
and key findings derived from this research have been succinctly encapsulated within the extended abstract paper, which has been graciously accepted for presentation 
at the [DMLR Workshop](https://dmlr.ai//). Nonetheless, I enthusiastically invite you to delve into the entirety of the thesis, where we have presented various extensions
and refinements of this project.

## Abstract 

Contrastive self-supervised learning has significantly improved the performance of deep learning methods, such as representation learning and clustering. However, due to their dependence on data augmentation, these methods are mostly utilized in computer vision. In this paper, we investigate the adaptation of the recent contrastive clustering approach in the case of tabular data. We utilize three-component architecture consists of backbone network, instance-level contrastive head and cluster-level contrastive head. The backbone network is responsible for generating meaningful latent vectors and its structure is optimized during fine-tuning process for each individual dataset. On the other hand, contrastive heads enable the model to concurrently learn about instance representations and cluster assignments. For contrastive learning, we introduce three augmentation techniques. Our experiments show that the model outperforms typical clustering methods applicable to tabular data in most cases. We underline the necessity of dataset-spectic augmentations and hyperparameters tuning. Furthermore, we test our model through various alterations, such as changing the contrastive loss or the method of incorporating a mask during augmentations. Our findings affirm the potential adaptability of successful contrastive clustering techniques from other fields, such as image processing, to the realm of tabular data.
![Image](/images/augmentations.png)
## Master thesis download:
[Master thesis](https://www.ap.uj.edu.pl/diplomas/167107/)

## Data-centric Machine Learning Research (DMLR) Workshop at ICML 2023 
[DMLR extended abstract paper](https://dmlr.ai/assets/accepted-papers/32/CameraReady/Contrastive_clustering___workshop-5.pdf)
