# GSCAN
GSCAN: Graph Stability Clustering using Edge-Aware Excess-of-Mass
##### Etzion Harari, Naphtali Abudarham & Roee Litman

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gscan-graph-stability-clustering-for/graph-clustering-on-citeseer)](https://paperswithcode.com/sota/graph-clustering-on-citeseer?p=gscan-graph-stability-clustering-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gscan-graph-stability-clustering-for/graph-clustering-on-cora)](https://paperswithcode.com/sota/graph-clustering-on-cora?p=gscan-graph-stability-clustering-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gscan-graph-stability-clustering-for/graph-clustering-on-pubmed)](https://paperswithcode.com/sota/graph-clustering-on-pubmed?p=gscan-graph-stability-clustering-for)

You can find full code example of using GSCAN for graph-clustering in this [**Google Colab Notebook**](https://colab.research.google.com/drive/1NZjeNUK_19OcFc29zq2mE7IGmBz_ZHy8#scrollTo=vwkQ-sxDJjdt).

![GSCAN_example](https://github.com/GraphEoM/GSCAN/blob/main/pictures/view.png)
**Two dimensional embeddings of the Cora dataset Planetoid using PCA, colored by clustering results of the three flavours of GSCAN.**

(a) GSCAN without a post process; One can notice the abundance of black points, which indicate data points that are flagged as outliers.

(b) GSCAN with intrinsic diffusion post-process; Here, the only remaining outliers are the ones that have no connectivity to any of the clusters.

(c) GSCAN with GNN-expansion post-process; This result has no outliers remaining. 

## Table of contents
1. [GSCAN Paper](https://github.com/GraphEoM/GSCAN#GSCAN-Paper)
2. [Install GSCAN](https://github.com/GraphEoM/GSCAN#Install-GSCAN)
3. [Dependencies for installation](https://github.com/GraphEoM/GSCAN#dependencies-for-installation)
4. [Run GSCAN](https://github.com/GraphEoM/GSCAN#run-gscan)
5. [Compare GSCAN results to KMeans](https://github.com/GraphEoM/GSCAN/blob/main/README.md#compare-gscan-results-to-kmeans-based-GNN-algorithm)

## GSCAN Paper
Our full paper available here: [GSCAN Paper](https://proceedings.mlr.press/v231/harari24a/harari24a.pdf)

The paper presented in the [LOG 2023 Conference](https://logconference.org/) (Learning on Graphs Conference) as a [poster](https://openreview.net/group?id=logconference.io/LOG/2023/Conference#tab-accept-poster) and published in [**PMLR**](https://proceedings.mlr.press/v231/harari24a.html):

![GSCAN_Poster](https://github.com/GraphEoM/GSCAN/blob/main/pictures/GSCAN_Poster.jpg)

If you find **GSCAN** useful in your research, you can cite the following paper:

``` sh

@InProceedings{harari24gscan,
  title = 	 {GSCAN: Graph Stability Clustering for Applications With Noise Using Edge-Aware Excess-of-Mass},
  author =       {Harari, Etzion and Abudarham, Naphtali and Litman, Roee},
  booktitle = 	 {Proceedings of the Second Learning on Graphs Conference},
  pages = 	 {9:1--9:15},
  year = 	 {2024},
  volume = 	 {231},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {27--30 Nov},
  publisher =    {PMLR}
}

```

## Install GSCAN
To install GSCAN on your device, you can clone this repo and pip install the [**wheel file**](https://github.com/GraphEoM/GSCAN/blob/main/dist/gscan-0.1.0-py3-none-any.whl) :

``` sh
git clone https://github.com/GraphEoM/GSCAN.git
cd GSCAN/dist
pip install gscan-0.1.0-py3-none-any.whl
```

After the installation, you can simply import GSCAN:
``` sh
from gscan import GSCAN
```

## Dependencies for installation

Torch and Torch-Geometric (for Linux):
``` sh
# for CPU:
pip install torch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# for GPU
pip install torch==1.12.1+cu113  -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

Others:
``` sh
umap_learn>=0.5.3
networkx>=2.8.8
numpy>=1.23.5
```
## Run GSCAN

#### load libraries
``` sh
import numpy as np
from gscan import GSCAN
```

#### load data
``` sh
# import libraries
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.datasets as db

# load data
data = db.Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures()).data

# get x,y and edges
features  = data.x
edges     = data.edge_index
labels    = data.y

# print dims
features.shape, edges.shape, labels.shape
```
(torch.Size([2708, 1433]), torch.Size([2, 10556]), torch.Size([2708]))


#### nodes features matrix
``` sh
features
```
| 0.14 | 0.98 | 0.77 |...| 0.81 | 0.96 | 0.76 |
|:----:|:----:|:----:|:--:|:----:|:----:|:----:|
| 0.73 | 0.7 | 0.96 |...| 0.64 | 0.0 | 0.95 |
| 0.83 | 0.62 | 0.5 |...| 0.24 | 0.51 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... |
| 0.39 | 0.37 | 0.16 |...| 0.86 | 0.65 | 0.26 |
| 0.84 | 0.52 | 0.96 |...| 0.08 | 0.2 | 0.21 |
| 0.45 | 0.27 | 0.45 |...| 0.48 | 0.37 | 0.22 |


#### edges table
``` sh
edges
```
| 29 | 14 | 26 | 29 | 21 | ... | 12 | 16 | 15 | 29 | 20 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 11 | 12 | 2 | 23 | ... | 6 | 13 | 5 | 24 | 4 |

#### activate GSCAN
``` sh
model = GSCAN(min_cluster_size=75)
model
```
<gscan.GSCAN at 0x1691c0d31c0>

#### Fit model
(if using_gae = True, GSCAN use GAE for learning the representation. if False, using just the original features matrix)
``` sh
model.fit(features,edges,using_gae=False)
```

#### Labels statistics (Vanilla GSCAN)
``` sh
np.unique(model.labels_,return_counts=True)
```
(array([-1,  0,  1,  2,  3,  4,  5,  6]),

 array([642, 384, 123, 198, 420, 191, 465, 285], dtype=int64))


#### Labels statistics (GSCAN + Intrinsic Diffusion)
``` sh
diffused_labels = model.diffuse_labels()

np.unique(diffused_labels,return_counts=True)
``` 
(array([-1,  0,  1,  2,  3,  4,  5,  6]),

 array([223, 457, 147, 208, 454, 219, 613, 387], dtype=int64))


#### Labels statistics (GSCAN + GNN Expansion)

``` sh
gnn_labels = model.gnn_labels(features,edges)
np.unique(gnn_labels,return_counts=True)
```
(array([0, 1, 2, 3, 4, 5, 6]),

 array([600, 248, 211, 427, 202, 637, 383], dtype=int64))

#### get the MST graph
``` sh
gnx = model.to_nx()
gnx
```
<networkx.classes.graph.Graph at 0x1c38c4db1c0>

## Compare GSCAN results to DAEGC (KMeans based GNN algorithm)
applied on DAEGC output representation, created using this code: [DAEGC](https://github.com/Tiger101010/DAEGC/blob/main/DAEGC/pretrain.py).
This repr. created by GNN clustering algorithm that published in 2019 in this paper:

[**Attributed Graph Clustering: A Deep Attentional Embedding Approach**](https://arxiv.org/abs/1906.06532)

It using GAE with both reconstruction loss and KL loss to learn latent representation to each node. The final step of this algorithm is to applying the KMeans on the output representation. We want to compare the results of KMeans on DAEGC output to the results of GSCAN.

#### Load DAEGC output representation
``` sh
import torch

features = torch.from_numpy(np.load(r'data\DAEGC.npy')) # pre-calcualted version of DAEGC output repr.
features.shape
```
torch.Size([2708, 16])

#### Activate & Fit GSCAN
``` sh
model = GSCAN(min_cluster_size=75).fit(features,edges,using_gae=False)
model
```
<gscan.GSCAN at 0x2224c139580>

#### Evaluate results

#### DAEGC (KMeans based) 
``` sh
from sklearn.cluster import KMeans
from gscan.evaluation import evl

cluster_labels = KMeans(n_clusters=7).fit(features).labels_
evl(labels,cluster_labels)
```
{'F1': 0.678024501465173,
 'ARI': 0.4383555248258824, 
 'NMI': 0.4869760432435305}

#### Vanilla GSCAN
``` sh
evl(labels,model.labels_)
```
{'F1': 0.6405708370997986,
 'ARI': 0.31377142336319763,
 'NMI': 0.5135693101580843}

#### GSCAN + Intrinsic Diffusion
``` sh
evl(labels,model.diffuse_labels())
```
{'F1': 0.7225066525853674,
 'ARI': 0.47756092195622213,
 'NMI': 0.5253690249400075}

#### GSCAN + GNN Expansion
``` sh
evl(labels,model.gnn_labels(features,edges))
```
{'F1': 0.7302434721526802, 
 'ARI': 0.503795736798374, 
 'NMI': 0.5340822319262325}
 
#### The same results in table format:

| Method | F1 | ARI | NMI |
|:---|:---:|:---:|:---:|
| DAEGC (KMeans based) | 0.678 | 0.438 | 0.487 |
| GSCAN | 0.640 | 0.314 | 0.513 |
| GSCAN + Intrinsic Diffusion | 0.722 | 0.477 | 0.525 |
| GSCAN + GNN Expansion | **0.730** | **0.503** | **0.534** |

## License
MIT
