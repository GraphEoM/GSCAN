# GSCAN
GSCAN: Graph Stability Clustering using Edge-Aware Excess-of-Mass

# Table of contents
1. [Dependencies installation](https://github.com/GraphEoM/GSCAN#dependencies-installation)
2. [Run GSCAN](https://github.com/GraphEoM/GSCAN#run-gscan)
3. [Compare GSCAN results to KMeans](https://github.com/GraphEoM/GSCAN/blob/main/README.md#compare-gscan-results-to-kmeans)

## Dependencies installation

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
(if using_gae = True, GSCAN use GAE for learning the representation. if False, the repr. using just the original features matrix)
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

## Compare GSCAN results to KMeans
applied on DAEGC output representation, created using this code: [DAEGC](https://github.com/Tiger101010/DAEGC/blob/main/DAEGC/pretrain.py).

#### Load DAEGC output representation
``` sh
import torch

features = torch.from_numpy(np.load(r'data\DAEGC.npy')) # pre-calcualted version of DAEGC output repr.
features.shape
```
torch.Size([2708, 16])

#### Activate & Fit GSCAN
``` sh
model = GSCAN(min_cluster_size=75).fit(nodes,edges,using_gae=False)
model
```

#### Evaluate results

#### KMeans 
``` sh
from sklearn.cluster import KMeans
from gscan.evaluation import evl

cluster_labels = KMeans(n_clusters=7).fit(nodes).labels_
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
evl(labels,model.gnn_labels(nodes,edges))
```
{'F1': 0.7302434721526802, 
 'ARI': 0.503795736798374, 
 'NMI': 0.5340822319262325}
 
#### Results in table:

| Method | F1 | ARI | NMI |
|:---:|:---:|:---:|:---:|
| KMeans | 0.678 | 0.438 | 0.487 |
| GSCAN | 0.640 | 0.314 | 0.513 |
| GSCAN + Intrinsic Diffusion | 0.722 | 0.477 | 0.525 |
| GSCAN + GNN Expansion | **0.730** | **0.503** | **0.534** |
