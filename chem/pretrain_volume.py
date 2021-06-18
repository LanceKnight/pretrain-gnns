import torch
import numpy as np
from data_generator import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GlobalAttention
from model import GNN
from torch.nn import MSELoss
import torch.optim as optim


class mGNN(torch.nn.Module):
    # modified version. All conv layers share the same parameters

    def __init__(self, gnn, num_layer, emb_dim, JK="last", drop_ratio=0, ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        # self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding = torch.nn.Linear(6, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.gnn = gnn

        for layer in range(num_layer):

            self.gnns.append(gnn)

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # self.x_embedding1(x[:, 0:9]) + self.x_embedding2(x[:, -1])
        x = self.x_embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # print(f'batch_norm:{self.batch_norms}, layer:{layer}')
            # print(f'h:{h.shape}')
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio,
                              training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GraphPool(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GraphPool, self).__init__()
        self.num_layer = num_layer
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type='gin')

        # if gnn_type == "gin":
        #     self.gnn = GINConv(emb_dim, aggr="add")
        # elif gnn_type == "gcn":
        #     self.gnn = GCNConv(emb_dim)
        # elif gnn_type == "gat":
        #     self.gnn = GATConv(emb_dim)
        # elif gnn_type == "graphsage":
        #     self.gnn = GraphSAGEConv(emb_dim)

        # self.gnn = mGNN(num_layer, self.gnn, emb_dim, JK, drop_ratio)
        # self.gnn_3 = GNN(3, self.gnn, emb_dim, JK, drop_ratio)

        self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))

    def save_weights(self, path):
        torch.save(self.gnn.state_dict(), path)

    def forward(self, data):
        node_rep = self.gnn(data)
        graph_rep = self.pool(node_rep, data.batch)
        return graph_rep


class PredictVol(torch.nn.Module):
    """docstring for PredictVol"""

    def __init__(self, input_dim, output_dim):
        super(PredictVol, self).__init__()

        self.graph_pred_linear = torch.nn.Linear(
            input_dim, output_dim)

    def forward(self, graph_rep):
        pred = self.graph_pred_linear(graph_rep)
        return pred


criterion = MSELoss()


def train(model_list, optimizer_list, loader, device):
    graph_model, pred_model = model_list
    graph_optimizer, pred_optimizer = optimizer_list

    model.train()
    pred_vol.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        graph_rep = graph_model(batch)
        pred = pred_model(graph_rep)
        loss = criterion(pred, torch.tensor(batch.volume))

        graph_optimizer.zero_grad()
        pred_optimizer.zero_grad()

        loss.backward()

        graph_optimizer.step()
        pred_optimizer.step()


def main():
    num_layer = 2
    emb_dim = 300
    JK = 'last'
    dropout_ratio = 0
    gnn_type = 'GIN'
    lr = 0.001
    decay = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    graph_model = GraphPool(num_layer, emb_dim, num_tasks=1, JK=JK, drop_ratio=dropout_ratio,
                            gnn_type=gnn_type, graph_pooling='attention').to(device)
    pred_vol = PredictVol(emb_dim, 1)

    model_list = [graph_model, pred_vol]

    optimizer_graph = optim.Adam(
        graph_model.parameters(), lr=lr, weight_decay=decay)
    optimizer_pred = optim.Adam(
        pred_vol.parameters(), lr=lr, weight_decay=decay)

    optimizer_list = [optimizer_graph, optimizer_pred]

    dataset = MoleculeDataset(
        root='dataset/pcba', molecular_weight=(0, 50))
    loader = DataLoader(dataset, batch_size=64)

    train(model_list, optimizer_list, loader, device)


if __name__ == '__main__':
    print('testing...')

    main()
