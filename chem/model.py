import torch
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.data import Data, DataLoader

from datetime import datetime

from kernels import PredefinedKernelSetConv, KernelSetConv
from loader import MoleculeDataset

import time


class MolGCN(MessagePassing):
    def __init__(self, num_layers=5, num_kernel1_1hop=0, num_kernel2_1hop=0, num_kernel3_1hop=0, num_kernel4_1hop=0, num_kernel1_Nhop=0, num_kernel2_Nhop=0, num_kernel3_Nhop=0, num_kernel4_Nhop=0, predefined_kernelsets=True, x_dim=5, p_dim=3, edge_attr_dim=1, ):
        super(MolGCN, self).__init__(aggr='add')
        self.num_layers = num_layers
        if num_layers < 1:
            raise Exception('at least one convolution layer is needed')

        self.layers = ModuleList()

        self.num_kernels_list = []
        # first layer
        if (num_kernel1_1hop is not None) and (num_kernel2_1hop is not None) and (num_kernel3_1hop is not None) and (num_kernel4_1hop is not None) and (predefined_kernelsets == False):
            kernel_layer = KernelSetConv(num_kernel1_1hop, num_kernel2_1hop, num_kernel3_1hop, num_kernel4_1hop, D=p_dim, node_attr_dim=x_dim, edge_attr_dim=edge_attr_dim)
            num_kernels = num_kernel1_1hop + num_kernel2_1hop + num_kernel3_1hop + num_kernel4_1hop
        elif (predefined_kernelsets == True):
            kernel_layer = PredefinedKernelSetConv(D=p_dim, node_attr_dim=x_dim, edge_attr_dim=edge_attr_dim, L1=num_kernel1_1hop,
                                                   L2=num_kernel2_1hop, L3=num_kernel3_1hop, L4=num_kernel4_1hop, is_first_layer=True)
            num_kernels = kernel_layer.get_num_kernel()
        else:
            raise Exception('MolGCN: num_kernel1-4 need to be specified')
        self.layers.append(kernel_layer)
        self.num_kernels_list.append(num_kernels)  # num of kernels in each layer

        # N layer
        x_dim = num_kernels
        for i in range(num_layers - 1):
            # print(f'layer:{i}')
            if (predefined_kernelsets == True):
                # print(f'num_kernels:{self.num_kernels(i)}')
                kernel_layer = PredefinedKernelSetConv(D=p_dim, node_attr_dim=self.num_kernels(i), edge_attr_dim=edge_attr_dim,
                                                       L1=num_kernel1_Nhop, L2=num_kernel2_Nhop, L3=num_kernel3_Nhop, L4=num_kernel4_Nhop, is_first_layer=False)
            else:
                kernel_layer = KernelSetConv(L1=num_kernel1_Nhop, L2=num_kernel2_Nhop, L3=num_kernel3_Nhop, L4=num_kernel4_Nhop,
                                             D=p_dim, node_attr_dim=self.num_kernels(i), edge_attr_dim=edge_attr_dim)
            self.layers.append(kernel_layer)
            self.num_kernels_list.append(kernel_layer.get_num_kernel())

    def num_kernels(self, layer):
        return self.num_kernels_list[layer]

    def forward(self, *argv, **kwargv):
        if len(argv) != 0:
            raise Exception('Kernel does not take positional argument, use keyword argument instead. e.g. model(data=data)')

        if len(kwargv) == 2:
            x = kwargv['data'].x
            edge_index = kwargv['data'].edge_index
            edge_attr = kwargv['data'].edge_attr
            p = kwargv['data'].p
            x_focal_deg1 = kwargv['data'].x_focal_deg1
            p_focal_deg1 = kwargv['data'].p_focal_deg1
            nei_x_deg1 = kwargv['data'].nei_x_deg1
            nei_p_deg1 = kwargv['data'].nei_p_deg1
            nei_edge_attr_deg1 = kwargv['data'].nei_edge_attr_deg1
            selected_index_deg1 = kwargv['data'].selected_index_deg1
            nei_index_deg1 = kwargv['data'].nei_index_deg1

            x_focal_deg2 = kwargv['data'].x_focal_deg2
            p_focal_deg2 = kwargv['data'].p_focal_deg2
            nei_x_deg2 = kwargv['data'].nei_x_deg2
            nei_p_deg2 = kwargv['data'].nei_p_deg2
            nei_edge_attr_deg2 = kwargv['data'].nei_edge_attr_deg2
            selected_index_deg2 = kwargv['data'].selected_index_deg2
            nei_index_deg2 = kwargv['data'].nei_index_deg2

            x_focal_deg3 = kwargv['data'].x_focal_deg3
            p_focal_deg3 = kwargv['data'].p_focal_deg3
            nei_x_deg3 = kwargv['data'].nei_x_deg3
            nei_p_deg3 = kwargv['data'].nei_p_deg3
            nei_edge_attr_deg3 = kwargv['data'].nei_edge_attr_deg3
            selected_index_deg3 = kwargv['data'].selected_index_deg3
            nei_index_deg3 = kwargv['data'].nei_index_deg3

            x_focal_deg4 = kwargv['data'].x_focal_deg4
            p_focal_deg4 = kwargv['data'].p_focal_deg4
            nei_x_deg4 = kwargv['data'].nei_x_deg4
            nei_p_deg4 = kwargv['data'].nei_p_deg4
            nei_edge_attr_deg4 = kwargv['data'].nei_edge_attr_deg4
            selected_index_deg4 = kwargv['data'].selected_index_deg4
            nei_index_deg4 = kwargv['data'].nei_index_deg4

            save_score = kwargv['save_score']
        else:
            x = kwargv['x']
            edge_index = kwargv['edge_index']
            edge_attr = kwargv['edge_attr']
            p = kwargv['p']

            x_focal_deg1 = kwargv['x_focal_deg1']
            p_focal_deg1 = kwargv['p_focal_deg1']
            nei_x_deg1 = kwargv['nei_x_deg1']
            nei_p_deg1 = kwargv['nei_p_deg1']
            nei_edge_attr_deg1 = kwargv['nei_edge_attr_deg1']
            selected_index_deg1 = kwargv['selected_index_deg1']
            nei_index_deg1 = kwargv['nei_index_deg1']

            x_focal_deg2 = kwargv['x_focal_deg2']
            p_focal_deg2 = kwargv['p_focal_deg2']
            nei_x_deg2 = kwargv['nei_x_deg2']
            nei_p_deg2 = kwargv['nei_p_deg2']
            nei_edge_attr_deg2 = kwargv['nei_edge_attr_deg2']
            selected_index_deg2 = kwargv['selected_index_deg2']
            nei_index_deg2 = kwargv['nei_index_deg2']

            x_focal_deg3 = kwargv['x_focal_deg3']
            p_focal_deg3 = kwargv['p_focal_deg3']
            nei_x_deg3 = kwargv['nei_x_deg3']
            nei_p_deg3 = kwargv['nei_p_deg3']
            nei_edge_attr_deg3 = kwargv['nei_edge_attr_deg3']
            selected_index_deg3 = kwargv['selected_index_deg3']
            nei_index_deg3 = kwargv['nei_index_deg3']

            x_focal_deg4 = kwargv['x_focal_deg4']
            p_focal_deg4 = kwargv['p_focal_deg4']
            nei_x_deg4 = kwargv['nei_x_deg4']
            nei_p_deg4 = kwargv['nei_p_deg4']
            nei_edge_attr_deg4 = kwargv['nei_edge_attr_deg4']
            selected_index_deg4 = kwargv['selected_index_deg4']
            nei_index_deg4 = kwargv['nei_index_deg4']

            data = Data(x=x, p=p, edge_index=edge_index, edge_attr=edge_attr,
                        x_focal_deg1=x_focal_deg1, x_focal_deg2=x_focal_deg2, x_focal_deg3=x_focal_deg3, x_focal_deg4=x_focal_deg4,
                        p_focal_deg1=p_focal_deg1, p_focal_deg2=p_focal_deg2, p_focal_deg3=p_focal_deg3, p_focal_deg4=p_focal_deg4,
                        nei_x_deg1=nei_x_deg1, nei_x_deg2=nei_x_deg2, nei_x_deg3=nei_x_deg3, nei_x_deg4=nei_x_deg4,
                        nei_p_deg1=nei_p_deg1, nei_p_deg2=nei_p_deg2, nei_p_deg3=nei_p_deg3, nei_p_deg4=nei_p_deg4,
                        nei_edge_attr_deg1=nei_edge_attr_deg1, nei_edge_attr_deg2=nei_edge_attr_deg2, nei_edge_attr_deg3=nei_edge_attr_deg3, nei_edge_attr_deg4=nei_edge_attr_deg4,
                        selected_index_deg1=selected_index_deg1, selected_index_deg2=selected_index_deg2, selected_index_deg3=selected_index_deg3, selected_index_deg4=selected_index_deg4,
                        nei_index_deg1=nei_index_deg1, nei_index_deg2=nei_index_deg2, nei_index_deg3=nei_index_deg3, nei_index_deg4=nei_index_deg4
                        )
            # print(f'foward: data.x{data.x}')
            save_score = kwargv['save_score']
        h = x

        for i in range(self.num_layers):
            print(f'{i}th layer')
            start = time.time()
            data.x = h

            kernel_layer = self.layers[i]
            sim_sc = kernel_layer(data=data, save_score=save_score)
            # print(f'edge_index:{edge_index.device}, sim_sc:{sim_sc.device}')
            # print('sim_sc')
            # print(sim_sc)
            h = self.propagate(edge_index=edge_index, sim_sc=sim_sc)
            end = time.time()
            print(f'layer time:{end-start}')
        return h

    def message(self, sim_sc_j):
        # print(f'sim_sc_j:{sim_sc_j.shape}')
        return sim_sc_j


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
            num_layer (int): the number of GNN layers
            emb_dim (int): dimensionality of embeddings
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            graph_pooling (str): sum, mean, max, attention, set2set
            gnn_type: gin, gcn, graphsage, gat
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layers=1, num_kernel1_1hop=0, num_kernel2_1hop=0, num_kernel3_1hop=0, num_kernel4_1hop=0, num_kernel1_Nhop=0, num_kernel2_Nhop=0, num_kernel3_Nhop=0, num_kernel4_Nhop=0, predefined_kernelsets=True, x_dim=5, p_dim=3, edge_attr_dim=1, JK="last", drop_ratio=0, graph_pooling="mean"):
        super(GNN_graphpred, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.D = p_dim

        if self.num_layers < 1:
            raise ValueError("GNN_graphpred: Number of GNN layers must be greater than 0.")

        self.gnn = MolGCN(num_layers=num_layers, num_kernel1_1hop=num_kernel1_1hop, num_kernel2_1hop=num_kernel2_1hop, num_kernel3_1hop=num_kernel3_1hop, num_kernel4_1hop=num_kernel4_1hop, num_kernel1_Nhop=num_kernel1_Nhop,
                          num_kernel2_Nhop=num_kernel2_Nhop, num_kernel3_Nhop=num_kernel3_Nhop, num_kernel4_Nhop=num_kernel4_Nhop, x_dim=x_dim, p_dim=p_dim, edge_attr_dim=edge_attr_dim, predefined_kernelsets=predefined_kernelsets)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        # elif graph_pooling == "attention":
        #     if self.JK == "concat":
        #         self.pool = GlobalAttention(gate_nn=torch.nn.Linear(
        #             (self.num_layer + 1) * emb_dim, 1))
        #     else:
        #         self.pool = GlobalAttention(
        #             gate_nn=torch.nn.Linear(emb_dim, 1))
        # elif graph_pooling[:-1] == "set2set":
        #     set2set_iter = int(graph_pooling[-1])
        #     if self.JK == "concat":
        #         self.pool = Set2Set((self.num_layer + 1)
        #                             * emb_dim, set2set_iter)
        #     else:
        #         self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layers + 1) * self.emb_dim, 1)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.gnn.num_kernels(-1), 1)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def save_kernellayer(self, path, time_stamp):
        layers = self.gnn.layers
        print(f'{self.D}D, there are {len(layers)} layers')
        for i, layer in enumerate(layers):
            print(f'saving {i}th layer')
            torch.save(layer.state_dict(), f'{path}/{time_stamp}_{i}th_layer.pth')

    def forward(self, *argv, save_score=False):
        if len(argv) == 33:
            x, p, edge_index, edge_attr, batch, \
                x_focal_deg1, x_focal_deg2, x_focal_deg3, x_focal_deg4,\
                p_focal_deg1, p_focal_deg2, p_focal_deg3, p_focal_deg4,\
                nei_x_deg1, nei_x_deg2, nei_x_deg3, nei_x_deg4,\
                nei_p_deg1, nei_p_deg2, nei_p_deg3, nei_p_deg4,\
                nei_edge_attr_deg1, nei_edge_attr_deg2, nei_edge_attr_deg3, nei_edge_attr_deg4,\
                selected_index_deg1, selected_index_deg2, selected_index_deg3, selected_index_deg4, \
                nei_index_deg1, nei_index_deg2, nei_index_deg3, nei_index_deg4\
                =\
                argv[0], argv[1], argv[2], argv[3], argv[4],\
                argv[5], argv[6], argv[7], argv[8], \
                argv[9], argv[10], argv[11], argv[12], \
                argv[13], argv[14], argv[15], argv[16],\
                argv[17], argv[18], argv[19], argv[20],\
                argv[21], argv[22], argv[23], argv[24],\
                argv[25], argv[26], argv[27], argv[28], \
                argv[29], argv[30], argv[31], argv[32],
        elif len(argv) == 1:
            data = argv[0]
            x, p, edge_index, edge_attr, batch,\
                x_focal_deg1, x_focal_deg2, x_focal_deg3, x_focal_deg4,\
                p_focal_deg1, p_focal_deg2, p_focal_deg3, p_focal_deg4,\
                nei_x_deg1, nei_x_deg2, nei_x_deg3, nei_x_deg4,\
                nei_p_deg1, nei_p_deg2, nei_p_deg3, nei_p_deg4,\
                nei_edge_attr_deg1, nei_edge_attr_deg2, nei_edge_attr_deg3, nei_edge_attr_deg4,\
                selected_index_deg1, selected_index_deg2, selected_index_deg3, selected_index_deg4,\
                nei_index_deg1, nei_index_deg2, nei_index_deg3, nei_index_deg4\
                =\
                data.x, data.p, data.edge_index, data.edge_attr, data.batch,\
                data.x_focal_deg1, data.x_focal_deg2, data.x_focal_deg3, data.x_focal_deg4,\
                data.p_focal_deg1, data.p_focal_deg2, data.p_focal_deg3, data.p_focal_deg4,\
                data.nei_x_deg1, data.nei_x_deg2, data.nei_x_deg3, data.nei_x_deg4,\
                data.nei_p_deg1, data.nei_p_deg2, data.nei_p_deg3, data.nei_p_deg4,\
                data.nei_edge_attr_deg1, data.nei_edge_attr_deg2, data.nei_edge_attr_deg3, data.nei_edge_attr_deg4,\
                data.selected_index_deg1, data.selected_index_deg2, data.selected_index_deg3, data.selected_index_deg4,\
                data.nei_index_deg1, data.nei_index_deg2, data.nei_index_deg3, data.nei_index_deg4
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x=x, edge_index=edge_index, edge_attr=edge_attr, p=p,
                                       x_focal_deg1=x_focal_deg1, x_focal_deg2=x_focal_deg2, x_focal_deg3=x_focal_deg3, x_focal_deg4=x_focal_deg4,
                                       p_focal_deg1=p_focal_deg1, p_focal_deg2=p_focal_deg2, p_focal_deg3=p_focal_deg3, p_focal_deg4=p_focal_deg4,
                                       nei_x_deg1=nei_x_deg1, nei_x_deg2=nei_x_deg2, nei_x_deg3=nei_x_deg3, nei_x_deg4=nei_x_deg4,
                                       nei_p_deg1=nei_p_deg1, nei_p_deg2=nei_p_deg2, nei_p_deg3=nei_p_deg3, nei_p_deg4=nei_p_deg4,
                                       nei_edge_attr_deg1=nei_edge_attr_deg1, nei_edge_attr_deg2=nei_edge_attr_deg2, nei_edge_attr_deg3=nei_edge_attr_deg3, nei_edge_attr_deg4=nei_edge_attr_deg4,
                                       selected_index_deg1=selected_index_deg1, selected_index_deg2=selected_index_deg2, selected_index_deg3=selected_index_deg3, selected_index_deg4=selected_index_deg4,
                                       nei_index_deg1=nei_index_deg1, nei_index_deg2=nei_index_deg2, nei_index_deg3=nei_index_deg3, nei_index_deg4=nei_index_deg4,
                                       save_score=save_score)
        # print(f'node_rep:{node_representation.shape}')
        graph_representation = self.pool(node_representation, batch)
        # print(f'graph_rep:{graph_representation.shape}')
        # print(f'linear layer shape:{self.graph_pred_linear}')
        pred = self.graph_pred_linear(graph_representation)
        # print(f'graph_rep:{graph_representation.shape}')
        # print(f'pred.grad:{pred.grad}')
        return pred, graph_representation


from tqdm import tqdm
import platform
import os

if __name__ == "__main__":
    D = 2
    dataset = '435008'
    # windows
    if (platform.system() == 'Windows'):
        root = 'D:/Documents/JupyterNotebook/GCN_property/pretrain-gnns/chem/dataset/'
    # linux
    else:
        root = '~/projects/GCN_Syn/examples/pretrain-gnns/chem/dataset/'
    if dataset == '435008':
        root = root + 'qsar_benchmark2015'
        dataset = dataset
    else:
        raise Exception('cannot find dataset')

    print(f'woring on {D}D now...')

    dataset = MoleculeDataset(D=D, root=root, dataset=dataset)
    dataset = dataset[234:236]

    # model = MolGCN(num_layers = 2, num_kernel_layers = 2, x_dim = 5, p_dim =3, edge_attr_dim = 1)
    model = GNN_graphpred(num_layers=5, num_kernel1_1hop=15, num_kernel2_1hop=15, num_kernel3_1hop=15, num_kernel4_1hop=15, num_kernel1_Nhop=15, num_kernel2_Nhop=15, num_kernel3_Nhop=15, num_kernel4_Nhop=15, x_dim=5, p_dim=D,
                          edge_attr_dim=1, JK='last', drop_ratio=0.5, graph_pooling='mean', predefined_kernelsets=False)
    model.save_kernellayer('saved_kernellayers')

    # loader = DataLoader(dataset, batch_size=2)
    # save_score = True
    # for data in loader:
    #     pred = model(data, save_score=save_score)
    #     print(f'pred:{pred}')
    #     save_score = False

    cri = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1):
        loader = DataLoader(dataset, batch_size=1)
        save_score = True
        for i, data in enumerate(loader):
            # print(f'=======data-{i}========')
            # print('-----before-----')
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'name:{name}, param:{param}, grad:{param.grad}')
            model.train()
            optimizer.zero_grad()
            pred, h = model(data)
            pred = pred.to(torch.float64)
            y = torch.ones(pred.shape).to(torch.float64)

            loss = cri(pred, y)
            print(f'loss:{loss}')
            loss.backward()

            # extra forward
            model.eval()
            pred, h = model(data)
            # print('---before update---')
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'name:{name}, param:{param}, grad:{param.grad}')
            # print(f'pred:{pred}, loss:{loss}, loss_grad:{loss.grad}')

            optimizer.step()
            # print('-----after update------')
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f'name:{name}, param:{param}, grad:{param.grad}')
            # print(f'pred:{pred}, loss:{loss}, loss_grad:{loss.grad}')
