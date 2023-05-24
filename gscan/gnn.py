## import libraries
# import torch objects
from torch.nn import Linear, ReLU, CrossEntropyLoss, Module, Sequential as Linear_Sequential
from torch_geometric.nn import Sequential as GCN_Sequential, GCNConv, GATConv
import numpy as np
import torch


# global variables
GCNLYR = 'GCN'
GATLYR = 'GAT'
LAYER  = '.'
TNSR   = type(torch.tensor([]))
PRC    = 'precision'
RCL    = 'recall'

# define useful function
torchify = lambda ary: ary if type(ary) == TNSR else torch.from_numpy(ary)
halo     = lambda s,c: [Halo(linewidth=s, foreground=c)]
softmax  = lambda m: m.exp()/torch.sum(m.exp(),dim=1).reshape(m.size()[0],1)
softmax  = torch.nn.Softmax(dim=1)

# GNN Layers
GNNS  = {GCNLYR: GCNConv,GATLYR: GATConv}
SIZE  = {GCNLYR: 1      ,GATLYR: 3}

# convert original dataframes to valid input graph
def covert_to_graph(nodes,edges,name,features,label):
    """
    convert to npy graph object
    :param nodes: data frame of the nodes feature + name
    :param edges: data frame of the edges link (name a to name b)
    :param name: the node name field name in the nodes df
    :param features: the columns of th feature in the nodes df
    :param label: the labels field name in the nodes df
    :return: nodes_features, edges_links, labels  vector
    """
    names      = [*nodes[name]]
    name_to_id = {names[i]:i for i in range(len(names))}
    edge_to_id = [(name_to_id[side_a],name_to_id[side_b]) for side_a,side_b in edges.values]
    return nodes[features].values.astype('float32'), np.array(edge_to_id).T, nodes[label].values

# Graph Convolutional Neural Network Object
class GNN(Module):
    """
    Torch GCN implementation
    """

    def __init__(self, conv_num=2, dense_num=2, epochs=100,
                 channels_num=16, dense_size=64,lr=.01,
                 print_status=True, type=GATLYR):
        """
        initilize GCN object
        :param conv_num: number of conv layers
        :param dense_num: number of densed layers
        :param channels_num: number of channels
        :param lr: learning rate
        :param epochs: number of epochs
        """
        super(GNN, self).__init__()

        # training parametres
        ## GNN layer type
        self.type       = type.upper()
        ## loss for each epoch
        self.loss       = []
        ## valid loss for each epoch
        self.valid_loss = []
        ## number of epochs
        self.epochs     = epochs
        ## learning rate
        self.lr         = lr
        ## print status while training
        self.print      = print_status

        # number of lyers from each type
        ## number of conv layers
        self.conv_num     = conv_num
        ## number of dense layers
        self.dense_num    = dense_num
        ## number of channels in each conv layer
        self.channels_num = channels_num
        ## number of neurons in each dense layer
        self.dense_size   = dense_size

        # Build the GCN
        ## device
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## optimizer
        self.optimizer  = None
        ## criterion
        self.criterion  = None

        ## build feature extractor
        self.extractor  = None
        ## build classifier
        self.classifier = None

    def build_extractor(self,feature_num):
        """
        builf feature conv extractor
        :param feature_num: number of input features
        :return: GCN extractor
        """
        layers = []
        for i in range(self.conv_num):
            # in each iteration ew add layer to our sequence
            input_size = feature_num if i == 0 else self.channels_num
            layers += [(GNNS[self.type](input_size, self.channels_num), 'x, edge_index -> x')]
            layers += [] if i + 1 == self.conv_num else [ReLU(inplace=True)]
        return GCN_Sequential('x, edge_index', layers)

    def build_classifier(self,labels_num):
        """
        build linear classifier
        :param labels_num: number of classes labels
        :return: linear classifier
        """
        layers = []
        for i in range(self.dense_num):
            # in each iteration ew add layer to our sequence
            last_layer  = i + 1 == self.dense_num
            input_size  = self.channels_num if i==0 else self.dense_size
            output_size = labels_num if last_layer else self.dense_size
            layers += [Linear(input_size, output_size)]
            layers += [] if last_layer else [ReLU(inplace=True)]
        return Linear_Sequential(*layers)

    def forward(self, x, edge_index):
        """
        forward function for define the network structure
        """
        x = self.extractor(x, edge_index)
        x = self.classifier(x)
        return x

    def build_network(self,input_size,output_size):
        """
        building the GCNetwork
        :param input_size:
        :param output_size:
        """
        # build layers
        self.extractor = self.build_extractor(input_size)
        self.classifier = self.build_classifier(output_size)

        # define optimizer & loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

        # activate GPU (if their is one)
        if torch.cuda.is_available():
            self = self.cuda()
            self.criterion = self.criterion.cuda()
        return self

    def fit(self, nodes, edges, labels, from_scratch=True,sample=[]):
        """
        training the GCN model
        :param nodes: input node features
        :param edges: input edges indicies table
        :param labels: input labels classes
        :param from_scratch: training the model from scratch (default: True)
        """
        # convert the numpy inputs to torch tensors
        nodes = torchify(nodes).to(self.device)
        edges = torchify(edges).long().to(self.device)
        labels = torchify(labels if len(sample)==0 else labels[sample]).long().to(self.device)

        # build GCNetwork
        if from_scratch:
            self.build_network(nodes.size()[1],labels.unique().size()[0])

        # training the network
        for epoch in range(self.epochs):

            # training
            self.train()
            self.optimizer.zero_grad()
            output = self(nodes, edges)

            # update weights
            loss = self.criterion(output if len(sample)==0 else output[sample], labels)
            loss.backward()
            self.optimizer.step()
            self.loss.append(loss.item())

            # plot status
            if epoch % 10 == 0 and self.print:
                print(f'epoch: {epoch}, loss: {self.loss[-1]}')

        return self

    def cross_fit(self, nodes, edges, labels,k=5):
        """
        k-folds cross validation fitting function
        :param nodes: input node features
        :param edges: input edges indicies table
        :param labels: input labels classes
        :param k: k folds (default: 5)
        :return: cross validation accuracy
        """
        accuracy = []
        folds = np.random.choice(np.arange(k),labels.shape[0])
        for i in range(k):
            # train model + prediction
            model = self.fit(nodes, edges, labels,sample=folds!=i)
            accuracy.append(model.score(nodes, edges, labels,sample=folds==i))
        return np.mean(accuracy)

    def valid_fit(self,nodes, edges, labels, from_scratch=True,sample=[]):
        """
        training the GCN model and check the validation set score
        :param nodes: input node features
        :param edges: input edges indicies table
        :param labels: input labels classes
        :param from_scratch: training the model from scratch (default: True)
        """
        # convert the numpy inputs to torch tensors
        nodes  = torchify(nodes)
        edges  = torchify(edges).long()
        labels = torchify(labels)

        # build GCNetwork
        if from_scratch:
            self.build_network(nodes.size()[1], labels.unique().size()[0])

        # training the network
        for epoch in range(self.epochs):

            # training
            self.train()
            self.optimizer.zero_grad()
            output = self(nodes, edges)

            # update weights
            loss = self.criterion(output[sample], labels[sample])
            loss.backward()
            self.optimizer.step()
            self.loss.append(loss.item())
            self.valid_loss.append(self.criterion(output[sample==False], labels[sample==False]))

            # plot status
            if epoch % 10 == 0 and self.print:
                print(f'epoch: {epoch}, loss: {self.loss[-1]}, valid: {self.valid_loss[-1]}')

        return self

    def project(self, nodes, edges):
        """
        use the convolutional extractor to get node emmbedding
        :param nodes: input nodes
        :param edges: input edges
        :return: node emmbedding in latent space
        """
        nodes = torchify(nodes).to(self.device)
        edges = torchify(edges).long().to(self.device)
        return self.extractor(nodes, edges).cpu().detach().numpy()

    def score(self, nodes, edges, labels,sample=[]):
        """
        calculate the accuracy of the network
        :param nodes: input node features
        :param edges: input edges indicies table
        :param labels: input labels classes
        :return: model accuracy score
        """
        sample = sample if len(sample)>0 else [True]*len(labels)
        return np.mean(self.predict(nodes, edges)[sample] == labels[sample])

    def predict(self, nodes, edges):
        """
        get prediction of the network
        :param nodes: input node features
        :param edges: input edges indicies table
        :return: prediction
        """
        nodes = torchify(nodes).to(self.device)
        edges = torchify(edges).long().to(self.device)
        with torch.no_grad():
            return np.argmax(torch.exp(self(nodes, edges)).cpu().numpy(), axis=1)

    def softvalue(self, nodes, edges):
        """
        get the certainy for each prediction
        :param nodes: input node features
        :param edges: input edges indicies table
        :return: certainy for each prediction
        """
        nodes = torchify(nodes)
        edges = torchify(edges).long()
        with torch.no_grad():
            output = torch.exp(self(nodes, edges)).cpu().numpy()
            return np.max(output, axis=1)/np.sum(output, axis=1)

    def get_soft(self,nodes,edges):
        """
        get the delta between 2 softvalues
        :param nodes: input node features
        :param edges: input edges indicies table
        :return: softmax delta values
        """
        nodes = torchify(nodes)
        edges = torchify(edges).long()
        with torch.no_grad():
            soft = softmax(self(nodes, edges))
            return torch.abs(soft[:,0]-soft[:,1]).numpy()

    def homophily(self,edges,labels):
        """
        calculate graph homophily
        :param edges: input edges indicies table
        :param labels: input labels classes
        :return: graph homophily
        """
        return np.mean(labels[edges[0,:]]==labels[edges[1,:]])

    def build_matrix(self, nodes, edges, labels, sample=[]):
        """
        build confusion matrix
        :param nodes: input node features
        :param edges: input edges indicies table
        :param labels: input labels classes
        :param sample: selected smaple from the data
        :return: confusion matrix
        """
        # build the empty matrix
        n      = np.unique(labels).shape[0]
        matrix = np.zeros(shape=(n, n))

        # get prediction
        predic = self.predict(nodes, edges)

        # filter by sample
        labels = labels[sample] if len(sample) else labels
        predic = predic[sample] if len(sample) else predic

        # add values to the matrix
        for i in range(labels.shape[0]):
            matrix[predic[i], labels[i]] += 1

        return matrix / labels.shape[0] * 100

    def metrics(self, nodes, edges, labels, sample=[]):
        """
        calculate precision & recall to every class
        :param nodes: input node features
        :param edges: input edges indicies table
        :param labels: input labels classes
        :param sample: selected smaple from the data
        :return: precision & recall to every class
        """
        matrix = self.build_matrix(nodes, edges, labels, sample)
        table = []
        for i in range(matrix.shape[0]):
            table.append({PRC: matrix[i][i] / matrix[i, :].sum(),
                          RCL: matrix[i][i] / matrix[:, i].sum()})
        return table

    def save(self,name):
        """
        save torch model as
        :param name: file name
        :return:
        """
        torch.save(self.state_dict(), f'{name}.pth')
        return self

    def load(self,name):
        """
        load torch model to the object
        :param name: model filename + path
        :return: model
        """
        loaded         = torch.load(f'{name}.pth')
        layers         = [*loaded.keys()]
        input_size     = loaded[layers[SIZE[self.type]]].size()[1]
        output_size    = loaded[layers[-1]].size()[0]

        self.build_network(input_size,output_size)
        self.load_state_dict(loaded,strict=False)
        return self

# MIT License