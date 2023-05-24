from torch_geometric.nn import GCNConv,GATConv
import torch

leaky_relu = lambda v,slope=.01: torch.where(v>0,v,slope*v)


class VGAEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GAAEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAAEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, cached=True,heads=1)
        self.conv2 = GATConv(2 * out_channels, out_channels, cached=True,heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def get_attention(self,x,edges):
        # weights = self.model.encoder.get_attention(x, edges)
        x = self.conv1(x, edges).relu()

        weights = [*self.parameters()]
        w_src = weights[-4][0][0]
        w_dst = weights[-3][0][0]
        betas = weights[-2]
        w_mat = weights[-1].T

        Ei, Ej = edges
        H = (x @ w_mat) + betas
        src = H @ w_src
        dst = H @ w_dst

        e = torch.exp(leaky_relu(src[Ei] + dst[Ej]))

        sum_e = torch.zeros(x.size()[0])
        for i in range(e.size()[0]):
            sum_e[Ei[i]] += e[i]

        return e/sum_e[Ei]

class GAEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


