# DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion

## 📋 Introduction
This is the implementation for the NeurIPS 2025 Conference paper 

_[DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](https://arxiv.org/abs/2507.11229)_.

_Jin Li_, _Zezhong Ding_, _Xike Xie_\*

### Abstract

Knowledge graphs (KGs) are vital for enabling knowledge reasoning across various domains. Recent KG reasoning methods that integrate both global and local information have achieved promising results. However, existing methods often suffer from **score over-smoothing**, which blurs the distinction between correct and incorrect answers and hinders reasoning effectiveness. To address this, we propose **DuetGraph**, a *coarse-to-fine* KG reasoning mechanism with *dual-pathway* global-local fusion. DuetGraph tackles over-smoothing by **segregating — rather than stacking —** the processing of local (via message passing) and global (via attention) information into two distinct pathways, preventing mutual interference and preserving representational discrimination. In addition, DuetGraph introduces a *coarse-to-fine* optimization, which partitions entities into high- and low-score subsets. This strategy narrows the candidate space and sharpens the score gap between the two subsets, thereby alleviating over-smoothing and enhancing inference quality. Extensive experiments on various datasets demonstrate that **DuetGraph achieves state-of-the-art (SOTA) performance**, with up to **8.7%** improvement in reasoning quality and a **1.8×** acceleration in training efficiency.


## 🚀 Getting Started

### Dependencies
- Python 3.9.21
- Pytorch 2.6.0
- CUDA 12.1
- pytorch-lightning 1.9.1
- torch-geometric 2.4.0
- torchmetrics 0.11.4
- einops 0.7.0
- numpy 1.24.1
- scipy 1.11.4
- scikit-learn 1.6.1
- tqdm 4.67.1


## 🌟 Citation
The paper is coming soon.




