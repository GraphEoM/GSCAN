# GSCAN: Graph Stability Clustering for Applications With Noise Using Edge-Aware Excess-of-Mass
# The full paper published in PLMR:
# https://proceedings.mlr.press/v231/harari24a.html

# created by Etzion Harari

# Version
__version__ =  0.9

# import libraries
from .models import VGAEncoder, GAEncoder, GAAEncoder
from torch_geometric.nn import VGAE, GAE
from umap import UMAP
from .gnn import GNN

from queue import PriorityQueue as PQ

import networkx as nx
import numpy as np
import torch

# useful functions
to_numpy  = lambda tensor: tensor.cpu().detach().numpy()
umap_2d   = lambda mat: torch.from_numpy(UMAP(n_components=2).fit_transform(to_numpy(mat)))
torchify  = lambda ary: ary if type(ary) == TNSR else torch.from_numpy(ary)
euclidean = lambda ei, ej: torch.sum(torch.pow(ei - ej, 2),dim=1).sqrt()
cosinesim = lambda ei, ej: 1-torch.sum(ei*ej,dim=1)/torch.sqrt(torch.sum(ei*ei,dim=1)*torch.sum(ej*ej,dim=1))


# global variables
CLUSTER_MERGE = 2
CLUSTER_GROWS = 1
EPSILON  = 1e-8
NOISE    = -1
TNSR     = type(torch.tensor([]))
MODELS   = {'GAE': lambda input_, output_:  GAE(GAEncoder( input_, output_)),
           'GAAE': lambda input_, output_:  GAE(GAAEncoder(input_, output_)),
           'VGAE': lambda input_, output_: VGAE(VGAEncoder(input_, output_))}
DISTANCE = {'euclidean': euclidean,
            'cosine':    cosinesim}

# cluster object for condensed tree
class Cluster:
    """
    Cluster object
    part of the condensed hierarchy
    """
    def __init__(self, name, size, index, weight, tree_size,right=None, left=None):
        """
        initilize cluster

        :param name: cluster id (int)
        :param size: cluster size - number of nodes (int)
        :param index: edge index (int)
        :param weight: edge weight (float)
        :param: tree_size: the current size of the tree
        :param right: right child cluster - large child cluster (Cluster object)
        :param left:  left  child cluster - small child cluster (Cluster object)
        """

        # sign true if this cluster is stable
        self.take = True

        # the cluster name
        self.name = name

        # the size of the cluster
        self.size = size

        # the first index of the cluster
        self.start = index

        # the last index of the cluster
        self.end = index

        # cluster indices
        self.indices = [index]

        # cluster weights
        self.weight = [weight]

        # cluster stability score
        self.stable = 0

        # the number of node that alreay in the condesend tree
        self.tree_size = tree_size

        # right child cluster - large child cluster (Cluster object)
        self.right = right

        # left  child cluster - small child cluster (Cluster object)
        self.left = left

        # did the cluster is a leaf
        self.leaf = True if right == None and left == None else False

    def __repr__(self):
        """
        represent a cluster object
        :return: string representation
        """

        # case of leaf cluster or branch
        if self.leaf:
            childs = 'type: leaf'
        else:
            childs = f'type: branch (right: {self.right.name}, left: {self.left.name})'

        return f'Cluster {self.name}, size: {self.size}, {childs}'

    def is_leaf(self):
        """
        if node is a leaf

        :return: True if this cluster is leaf, else False
        """
        return self.leaf

    def get_stability(self):
        """
        calculate the cluster stability
        if the stability pre-calculated - just return it

        :return: the cluster stability
        """
        if self.stable > 0:
            return self.stable
        else:
            # stability calculation using HDBSCAN Optimizer
            # based on:
            # "Density-based clustering based on hierarchical density estimates"
            #  Campello, Moulavi, Sander (2013)
            self.stable = sum([((1 / w) - (1 / self.weight[-1])) for w in self.weight])
            return self.stable

    def grow(self, index, weight,tree_size):
        """
        increase the cluster size

        :param index: new edge index for the cluster indices list
        :param: tree_size: the current size of the tree
        :param weight: new edge weight for the cluster weights list
        :return: None
        """
        self.tree_size = tree_size
        self.end       = index
        self.indices  += [index]
        self.weight   += [weight]
        self.size     += 1
        return None


class GSCAN:
    """
    GSCAN object

    graph clustering object

    Processing steps:
    1. calculate node embedding using GAE/GVAE.
    2. build MST where each edge weight defined by the embedding
    3. find the stable clusters in the MST
    4. extract clusters

    """
    def __init__(self, model='GAAE', print_by=5, epochs=50, metric='cosine',
                 min_cluster_size=5, dim_reduce=False, lr=0.01,
                 allow_single_cluster=False):
        """
        inilize GSCAN object

        :param model: type of GAE model (GAE/GVAE)
        :param print_by: number of epochs to each status print
        :param epochs: number of GAE epochs
        :param distance: distance metrics
        :param min_cluster_size: min cluster size
        :param dim_reduce: using UMAP to reduce Z dimension
        :param lr: learning rate for the GAE
        :param allow single cluster: allowing single cluster ouput (default: False)
        """

        # aloowing single cluster
        self.single     = allow_single_cluster

        # current processing device
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # the distance metric
        self.delta_f    = DISTANCE[metric] if metric in DISTANCE else metric

        # number of epochs
        self.epochs     = epochs

        # dimension reduction option
        self.reduce     = dim_reduce

        # model type (GAE/GVAE)
        self.model_type = model

        # number of epochs to each status print
        self.print_by   = print_by

        # min cluster size
        self.min_size   = min_cluster_size

        # learning rate
        self.lr         = lr


        # number of not tagged nodes
        self.tree_size  = 0

        # graph edges
        self.edges      = None

        # edges weights (length)
        self.length     = None

        # clusters dictionary
        self.clusters   = {}

        # labels (called labels_ because sklearn syntax)
        self.labels_    = None

        # edge status (in the MST or not)
        self.in_mst     = []

        # the GAE/GVAE model
        self.model      = None

        # noise labels (by the model results)
        self.noise      = None

        # model loss
        self.loss       = []

        # the mst
        self.mst        = None

        # the embedding representation
        self.Z          = None

        # number of nodes in the graph
        self.n          = None

    def embedding(self, features, edges, output_dim):
        """
        embedding the graph nodes, using GAE

        :param features: the graph feature
        :param edges: the graph edges
        :param output_dim: the embedding dimension
        :return: None
        """
        # build Graph Auto Encoder
        input_dim = features.size()[1]
        self.model = MODELS[self.model_type.upper()](input_dim, output_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # train network
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()

            # encoding decoding
            Z = self.model.encode(features, edges)
            loss = self.model.recon_loss(Z, edges)

            # loss and back-prop
            loss += (1 / self.n) * self.model.kl_loss() if self.model == 'VGAE' else 0
            loss.backward()
            optimizer.step()
            self.loss.append(loss.item())

            # print status
            if epoch % self.print_by == 0:
                print(f'epoch: {epoch}, loss: {self.loss[-1]}')

        # dim reduce
        self.Z = (umap_2d(Z) if self.reduce else Z).to(self.device)
        return None

    def attention_edges(self,features,edges):
        """
        calculate the attention of each edge
        (active when self.model_type=='GAAE')
        we get the attention from the last layer in the GAE

        :param features: features matrix
        :param edges: graph edges
        :return: alpha values (edges attention values)
        """

        if self.model_type.upper()!='GAAE':
            # Graph Auto Encoder not using Attention!
            # cannot calculate edge attention
            return False

        # get the final layer
        last_layer = self.model.encoder.conv2

        # calculate the output of the first layer
        X = self.model.encoder.conv1(features, edges).relu()

        # get WX for desination and source
        src_x = last_layer.lin_src(X).view(-1, 1, last_layer.out_channels)
        dst_x = last_layer.lin_dst(X).view(-1, 1, last_layer.out_channels)

        # find the alpha values for each side
        src_alpha = (src_x * last_layer.att_src).sum(dim=-1)
        dst_alpha = (dst_x * last_layer.att_dst).sum(dim=-1)
        alpha     = (src_alpha, dst_alpha)

        # caluclate the alpha softmax per-edge
        alpha = torch.exp(last_layer.edge_updater(edges, alpha=alpha, edge_attr=None))
        a_sum = torch.zeros(features.shape[0])
        for eij in range(edges.shape[1]):
            ei,_ = edges[:,eij]
            ai   = alpha[eij][0]
            a_sum[ei] += ai

        alpha = alpha.view(edges.shape[1])/(a_sum[edges[0,:]]+EPSILON)

        return 1/(alpha+EPSILON)

    def edges_by_attention(self,features,edges):
        """
        sort graph edges by their length (using GAT attention mechanism)

        :param features: features matrix
        :param edges: graph edges
        :return: sorted edges & weights
        """
        weights     = self.attention_edges(features,edges)
        sort_edges  = weights.sort()
        self.edges  = to_numpy(edges.T[sort_edges.indices, :])
        self.length = to_numpy(sort_edges.values)
        return None

    def edges_by_distance(self,edges):
        """
        sort graph edges by their length

        :param edges: graph edges
        :return: sorted edges & weights
        """
        weights     = self.delta_f(self.Z[edges[0, :]], self.Z[edges[1, :]])+EPSILON
        sort_edges  = weights.sort()
        self.edges  = to_numpy(edges.T[sort_edges.indices, :])
        self.length = to_numpy(torch.where(sort_edges.values>0,sort_edges.values,EPSILON))
        return None

    def find(self, index):
        """
        find index cluster

        :param index: given index
        :return: index cluster
        """
        while self.labels_[index] != index:
            index = self.labels_[index]
        return index

    def condensed_tree(self):
        """
        building MST & condensed tree
        using kruskal Algorithm and find+join data structure

        :return: None
        """

        self.clusters = {}
        self.in_mst   = []
        self.labels_  = [*range(self.n)]
        cluster_size  = [1] * self.n

        iter_num = -1

        for ei, ej in self.edges:
            # update iter index
            iter_num += 1

            # find the components of each edge
            com_i = self.find(ei)
            com_j = self.find(ej)

            # check if (i,j) in MST
            spanning_edge = com_i != com_j
            self.in_mst.append(spanning_edge)

            # if (i,j) in MST:
            if spanning_edge:

                # find which component bigger
                large_com = com_i if cluster_size[com_i] >= cluster_size[com_j] else com_j
                small_com = com_j if cluster_size[com_i] >= cluster_size[com_j] else com_i

                # join the small component to the big component
                self.labels_[small_com] = self.labels_[large_com]
                cluster_size[large_com] += cluster_size[small_com]

                # join status
                large_status = (cluster_size[large_com] >= self.min_size)
                small_status = (cluster_size[small_com] >= self.min_size)
                status = large_status + small_status

                # add edge epsilon to com-stability set
                if CLUSTER_GROWS == status:
                    if large_com in self.clusters:
                        self.tree_size += cluster_size[small_com]
                        self.clusters[large_com].grow(iter_num, self.length[iter_num],self.tree_size)
                    else:
                        self.tree_size += cluster_size[large_com]
                        self.clusters[large_com] = Cluster(name      = large_com,
                                                           size      = cluster_size[large_com],
                                                           index     = iter_num,
                                                           weight    = self.length[iter_num],
                                                           tree_size = self.tree_size)

                # cluster merged
                if CLUSTER_MERGE == status:
                    # merge the two clusters to new one
                    self.clusters[large_com] = Cluster(name      = large_com,
                                                       size      = cluster_size[large_com],
                                                       index     = iter_num,
                                                       weight    = self.length[iter_num],
                                                       tree_size = self.tree_size,
                                                       right     = self.clusters[large_com],
                                                       left      = self.clusters[small_com])
                    del self.clusters[small_com]

        # build mst
        self.mst = self.edges[self.in_mst, :]
        return None

    def stability_opitimizer(self, cluster):
        """
        clusters stability optimizer in the condensed tree
        sign the stable clusters in the condensed tree
        (recursive function)

        :param cluster: given cluster
        :return: cluster stability
        """

        # if the cluster is a leaf
        if cluster.is_leaf():
            return cluster.get_stability()
        else:
            # calculate the stability of the clusters childrens (right & left)
            right = self.stability_opitimizer(cluster.right)
            left  = self.stability_opitimizer(cluster.left)

            childs = right + left

            # if the stability of the childs is better than the cluster
            if childs > cluster.get_stability():
                cluster.stable = childs
                cluster.take = False
                return cluster.stable

            # if the stability of the clusters is better than of the childs
            else:
                cluster.right = None
                cluster.left = None
                cluster.leaf = True
                return cluster.stable

    def get_clusters(self, cluster, clusters, unstable):
        """
        extract cluster from the condensed
        (recursive function)

        :cluster: given cluster
        :clusters: optimal clusters dictionary
        :unstable: list of unstable edges
        """
        # if cluster is leaf, add it to optimal clusters
        if cluster.is_leaf():
            clusters[cluster.name] = cluster

        # if the cluster not optimal
        else:
            unstable += cluster.indices
            self.get_clusters(cluster.right, clusters, unstable)
            self.get_clusters(cluster.left,  clusters, unstable)

    def prevent_single_cluster(self):
        """
        if this function activate, the object prevent extraction
        of one single cluster, unless the min_cluster_size forcing
        such output

        :return: None
        """
        for cluster in self.clusters.values():
            cluster.stable = EPSILON

        return None

    def find_stable_clusters(self):
        """
        find the stable clusters
        and mark the unstable edges

        :return: unstable edges
        """
        unstable_edges = []
        stable_clusters = {}

        # handle with single cluster scenario
        if not self.single:
            self.prevent_single_cluster()

        # remove unstable leafs
        for cluster in self.clusters.values():
            self.stability_opitimizer(cluster)

        # choose only stable leafs
        for cluster in self.clusters.values():
            self.get_clusters(cluster, stable_clusters, unstable_edges)

        self.clusters = stable_clusters
        return set(unstable_edges)

    def extract_labels(self, unstable):
        """
        extract the labels of each cluster to its nodes

        :unstable: unstable edges
        """
        self.labels_ = [*range(self.n)]
        cluster_size = [1] * self.n

        # name of each cluster
        names = {ci: i for ci, i in zip(self.clusters, range(len(self.clusters)))}

        # calcualte the labels using the MST
        for iteration in range(len(self.in_mst)):

            # only if the edge is part of the MST
            if self.in_mst[iteration]:
                ei, ej = self.edges[iteration]

                com_i = self.find(ei)
                com_j = self.find(ej)

                # find which component bigger
                large_com = com_i if cluster_size[com_i] >= cluster_size[com_j] else com_j
                small_com = com_j if cluster_size[com_i] >= cluster_size[com_j] else com_i

                # pass if the edge belong to unstable cluster
                if iteration not in unstable:
                    self.labels_[small_com] = large_com
                    cluster_size[large_com] += cluster_size[small_com]

        # give each point labels
        self.labels_ = [self.find(i) for i in self.labels_]

        # filter small components
        self.labels_ = [names[i] if cluster_size[i]>=self.min_size else NOISE for i in self.labels_]

        # convert self.labels_ to np.array
        self.labels_ = np.array(self.labels_)

        # define which node is "noise"
        self.noise   = self.labels_==NOISE

        return None

    def cluster_by_density(self):
        """
        calculate clusters using the density estimation of each edge length

        :return: None
        """
        # build condensed tree + MST
        self.condensed_tree()

        # find stable clusters & unstable edges
        unstable_edges = self.find_stable_clusters()

        # extract clustering labels
        self.extract_labels(unstable_edges)

        return None

    def refit(self):
        """
        re-fit the clusters
        you can use this function after you made some changes
        in the parameters of the model,
        without need to train the GAA again

        :return: self
        """
        self.tree_size = 0
        self.cluster_by_density()

        return self

    def fit(self, features, edges, output_dim=64, using_gae=True):
        """
        fit the clustering model to the given data

        :param features: graph features
        :param edges: graph edges
        :param output_dim: embedding dimension
        :param using_gae: using GAE for node embedding
        :return: None
        """

        # pre-processing
        features = torchify(features).to(self.device)
        edges    = torchify(edges if edges.shape[0] == 2 else edges.T).to(self.device).long()
        self.n   = features.size()[0]

        # representation method
        if using_gae:
            # get node embedding using GAE
            self.embedding(features, edges, output_dim)
        else:
            # using the simple feature matrix
            self.Z = features

        # sort edges by length
        self.edges_by_distance(edges)

        # calculate the clusters using graph-density
        self.cluster_by_density()

        return self

    def to_nx(self):
        """
        export MST to networkx graph

        :return: networkx graph
        """
        G = nx.Graph()
        for (i,j),wij in zip(self.mst,self.length[self.in_mst]):
            G.add_edge(i,j, weight = wij)
        return G

    def get_nodes_weights(self,lambda_value=False):
        """
        get the weights of each node to each of the clusters
        in the graph (including "noise" cluster).

        :param lambda_value: using 1/weight insead just weight (default: False)
        :return: weights for each node to each cluster
        """

        # weights array by key (node,cluster)
        array_dct = {}

        # run on each edge to get its weights
        eij = 0
        for ei, ej in self.edges:

            # define variables
            key_ij     = (ei, self.labels_[ej])
            weight_ij  = [1/(self.length[eij]+EPSILON)] if lambda_value else [self.length[eij]]

            # add the edge weight to the dict
            if key_ij in array_dct:
                array_dct[key_ij]+= weight_ij
            else:
                array_dct[key_ij] = weight_ij

            # index for next edge weight
            eij += 1

        return array_dct

    def soft_clustering(self, softmax=False,aggfunc=np.max,include_noise=True):
        """
        soft clustering for each node in the graph

        this function gives each node the probability
        of being part in each cluster in the graph

        :param softmax: using softmax function (default: False)
        :param aggfunc: the aggregation function for the weights of each edge
        :param include_noise: include "noise cluster" column (default: False)
        :return: probabilites for each node of being in each cluster
        """

        # get the weights from each node to each cluster
        array_dct = self.get_nodes_weights(lambda_value=True)

        # caluclate the probabilites for each cell in the matrix
        probs = np.zeros((self.n, len(self.clusters) + 1))
        for i,j in array_dct:
            probs[i,j] = aggfunc(array_dct[(i,j)])

        # filter noise (optional)
        probs = probs if include_noise else probs[:,:NOISE]

        # using softmax for probabilities
        if softmax:
            probs = np.where(probs>0,np.exp(probs),0)

        # return probabilities with sum of one
        sum_probs = np.sum(probs, axis=1).reshape((self.n, 1))+EPSILON
        return probs/sum_probs

    def add_soft_labels(self, threshold=.1, softmax=False, aggfunc=np.max):
        """
        add cluster labels for noise nodes with high probability
        using soft clustering probabilities

        :param threshold: threshold to choose only noise nodes with high probability
        :param softmax: using softmax function (default: False)
        :param aggfunc: the aggregation function for the weights of each edge
        """

        # get soft clustering probabilities
        probs     = self.soft_clustering(softmax, aggfunc)

        # select only nodes with hight probabilities
        high_prob = np.max(probs[:, :NOISE], axis=1) >= threshold

        # select only noise nodes with hight probability
        choose    = (self.noise*high_prob)>0

        # add clusters labels for noise nodes
        self.labels_[choose] = np.argmax(probs[choose,:NOISE],axis=1)

        return None

    def restore_labels(self):
        """
        this function remove labels which gaved by the soft clustering,
        using the "add soft labels" above function. after using this function,
        each "noise" label that gave some cluster label, became -1 again.

        :return: None
        """
        self.labels_[self.noise] = NOISE

        return None

    def clusters_per_node(self,threshold=.1,softmax=False, aggfunc=np.max, include_noise=True):
        """
        find the number of potential neighbors of each node

        :param threshold: threshold to choose only noise nodes with high probability
        :param softmax: using softmax function (default: False)
        :param aggfunc: the aggregation function for the weights of each edge
        :param include_noise: include "noise cluster" column (default: False)
        :return: number of clusters per each node
        """
        # calculate soft probabilities for each node
        probs      = self.soft_clustering(softmax, aggfunc, include_noise)

        # calcualte the number of potential neighbors for each node
        partitions = np.sum((probs[:, :NOISE] if include_noise else probs) > threshold, axis=1)

        # remove value of 0 from vector (belong for Single nodes)
        partitions[partitions==0] = 1
        
        return partitions

    def diffuse_labels(self,mst=True):
        """
        using the MST to assign label for each noise-node in the graph.
        each noise node get the label of the nearest tagged node,
        where "nearest" means in terms of the distance across the MST.
        :param mst: diffuse the labels only in the MST (default: True)
        :return: labels after diffusion
        """

        # labels to edit
        labels   = [*self.labels_]

        # intrinsic distance to each node
        distance = np.zeros(len(labels))
        distance[self.labels_==-1] = np.inf

        # Priority Queue
        links    = PQ()

        # mst graph of the noise nodes
        G        = nx.Graph()

        # define the edges for the MST/entire Graph
        edges = zip(self.mst, self.length[self.in_mst]) if mst else zip(self.edges,self.length)

        # for each edge in the MST:
        for (i, j), wij in edges:

            i_noise = labels[i] < 0
            j_noise = labels[j] < 0
            status  = (i_noise * 1.0) + (j_noise * 1.0)

            # edges with label in one side
            if status == 1:
                labeled_i, noise_j = (i, j) if j_noise else (j, i)
                links.put((wij, (labeled_i, noise_j)))

            # edges with noise nodes (including with just one side)
            if status > 0:
                G.add_edge(i, j, weight=wij)

        # for each edge with label in one side:
        while not links.empty():

            # get edge indices
            wij, (i, j) = links.get()

            # if node j is indeed noise (and their short path to that node)
            if distance[j] > wij:

                # assign the new distance
                distance[j] = wij

                # assign label to node j
                labels[j] = labels[i]

                # for each neighbor of node j
                for neighbor in G.neighbors(j):

                    # key and weight to the new neighbor
                    key = (j, neighbor)
                    weight = G.edges[key]['weight']

                    # if the distance for the neighbor is the shortest (for now)
                    if distance[neighbor] > distance[j]+weight:

                        # add the neighbor to the links queue
                        links.put((weight+distance[j], key))

        return np.array(labels)

    def gnn_labels(self,features,edges,lr=0.01):
        """
        expand the GSCAN labels using GNN
        the gnn based on classic PyG classification model

        :param features: nodes features matrix
        :param edges: graph edges index
        :param lr: learning rate
        :return: gnn labels
        """
        # convert edges to proper input
        edges = torchify(edges if edges.shape[0] == 2 else edges.T).to(self.device).long()

        # copy current labels
        labels = np.array([*self.labels_])

        # activate GNN
        gnn = GNN(print_status=False,lr=lr)

        # training GNN model on labels (labels>=0)
        gnn.fit(features,edges,labels,sample=labels>=0)

        # replace noise labels with learned labels
        labels[labels==-1] = gnn.predict(features,edges)[labels==-1]

        return labels

# MIT License
# created by Etzion Harari
