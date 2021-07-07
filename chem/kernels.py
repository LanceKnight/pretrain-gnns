# test of kernel with L

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import degree

import torch
from torch.nn import ModuleList
from torch.nn import CosineSimilarity, Module, ModuleList, Linear, Sigmoid
from torch.nn.parameter import Parameter

from itertools import permutations
import math


class KernelConv(Module):
    def __init__(self, L=None, D=None, num_supports=None, node_attr_dim=None, edge_attr_dim=None, init_kernel: 'type(Data)'=None):
        super(KernelConv, self).__init__()
        if init_kernel is None:
            if (L is None) or (D is None) or (num_supports is None) or (node_attr_dim is None) or (edge_attr_dim is None):
                raise Exception(
                    'either numer of kernels L, convolution dimention D, number of support num_supports or feature dimension node_attr_dim is not specified')
            else:
                init_kernel = Data(x_center=torch.randn(L, 1, node_attr_dim),  x_support=torch.randn(
                    L, num_supports, node_attr_dim), edge_attr_support=torch.randn(L, num_supports, edge_attr_dim), p_support=torch.randn(L, num_supports, D))

        x_center_tensor = init_kernel.x_center
        self.x_center = Parameter(x_center_tensor)

        x_support_tensor = init_kernel.x_support
        self.x_support = Parameter(x_support_tensor)

        edge_attr_support_tensor = init_kernel.edge_attr_support
        self.edge_attr_support = Parameter(edge_attr_support_tensor)

        p_support_tensor = init_kernel.p_support
#         print(f'p_support_tensor:{p_support_tensor.shape}')
        self.p_support = Parameter(p_support_tensor)

    def permute(self, x):
        #         print('permute')
        #         print('x')
        #         print(x.shape)
        rows = x.shape[1]
        l = [x[:, torch.tensor(permute), :]
             for permute in list(permutations(range(rows)))]
        output = torch.stack(l, dim=1)
#         print('permuted')
#         print(output.shape)
        return output

    def intra_angle(self, p):
        '''
        angles between each row vectors
        '''
        cos = CosineSimilarity(dim=-1)
        new_p = torch.roll(p, 1, dims=-2)
#         print(f'new p:')
#         print(new_p)
        sc = cos(new_p, p)
#         print(f'intra angle sc:{sc.shape}')
        return sc

    def arctan_sc(self, tensor1, tensor2, dim=None):
        diff = torch.square(tensor1 - tensor2)
#         print(diff)
        if dim is not None:
            sc = torch.sum(diff, dim=dim)
        else:
            sc = torch.sum(diff)
        sc = torch.atan(1 / sc)
        return sc

    def get_angle_score(self, p_neighbor, p_support):

        #         p_neighbor = p_neighbor.unsqueeze(0).expand(p_support.shape[0], p_neighbor.shape[0],p_neighbor.shape[1],p_neighbor.shape[2])
        #         p_support = p_support.unsqueeze(1).expand(p_support.shape[0],p_neighbor.shape[1],p_support.shape[1],p_support.shape[2])

        #         print('get_angle_score')
        #         print('p_neighbor.shape')
        #         print(p_neighbor.shape)
        #         print('p_support.shape')
        #         print(p_support.shape)
        if(p_support.shape[2] == 1):
            return torch.full((p_support.shape[0], 1), math.pi / 2)
#         cos = CosineSimilarity(dim = 1)

        p_neighbor = p_neighbor.unsqueeze(0).unsqueeze(0).expand(
            p_support.shape[0], p_support.shape[1], p_neighbor.shape[-3], p_neighbor.shape[-2], p_neighbor.shape[-1])
#         p_neighbor = p_neighbor.unsqueeze(0).expand(p_support.shape)
        intra_p_neighbor_dist = self.intra_angle(p_neighbor)
#         intra_p_neighbor_dist = intra_p_neighbor_dist.unsqueeze(0).expand(p_support.shape[0], p_neighbor.shape[0],p_neighbor.shape[1])
#         print(f'intra_p_neighbor_dist:{intra_p_neighbor_dist.shape}')

        p_support = p_support.unsqueeze(2).expand(p_neighbor.shape)
#         print(f'p_support:{p_support.shape}')
        intra_p_support_dist = self.intra_angle(p_support)
#         intra_p_support_dist = intra_p_support_dist.unsqueeze(1).expand(p_support.shape[0], p_neighbor.shape[0],p_support.shape[1])
#         print(f'intra_p_support_dist:{intra_p_support_dist.shape}')

#         sc = cos(intra_p_neighbor_dist, intra_p_support_dist)
#         sc = torch.dot(intra_p_neighbor_dist, intra_p_support_dist.T)
        sc = self.arctan_sc(intra_p_neighbor_dist,
                            intra_p_support_dist, dim=(-1, -2))
#         print(f'angle_sc:{sc}')
        return sc

    def get_length_score(self, p_neighbor, p_support):
        len_p_neighbor = torch.norm(p_neighbor, dim=-1)
        len_p_support = torch.norm(p_support, dim=-1)

#         print('len_p_neighbor')
#         print(len_p_neighbor.shape)
#         print(len_p_support.shape)

        # inverse of L2 norm
        sc = self.arctan_sc(len_p_neighbor, len_p_support, dim=(-2, -1))
#         diff = torch.square(len_p_neighbor - len_p_support)
#         sc = torch.sum(diff)
#         sc = torch.atan(1/sc)

# #         print(sc)
        return sc

    def get_support_attribute_score(self, x_nei, x_support: 'shape(s, d)'):

        # inverse of L2 norm
        #         print('x_nei')
        #         print(x_nei)
        #         print('x_support')
        #         print(x_support)
        #         diff = torch.square(x_nei - x_support)
        #         sc = torch.sum(diff)
        #         sc:'shape([])' = torch.atan(1/sc)
        sc = self.arctan_sc(x_nei, x_support, dim=(-3, -2, -1))
        return sc

    def get_center_attribute_score(self, x_focal, x_center):
        # inverse of L2 norm
        #         print(f'center attri:{type(x_focal)}, {type(x_center)}')
        #         print(f'x_focal:{x_focal.shape}, x_center:{x_center.shape}')
        #         print(x_focal)
        #         print(x_center)
        #         diff = torch.square(x_focal - x_center)
        #         sc = torch.sum(diff)
        #         sc:'shape([])' = torch.atan(1/sc)
        x_focal = x_focal.unsqueeze(0).expand(
            x_center.shape[0], x_focal.shape[0], x_focal.shape[1])
        x_center = x_center.expand(x_focal.shape)
        sc = self.arctan_sc(x_focal, x_center, dim=(-2, -1))
        return sc

    def get_edge_attribute_score(self, edge_attr_nei, edge_attr_support):
        #         print('edge_attr_nei')
        #         print(edge_attr_nei)
        #         print('edge_attr_support')
        #         print(edge_attr_support)
        #         diff = torch.square(edge_attr_nei - edge_attr_support)
        #         sc = torch.sum(diff)
        #         sc:'shape([])' = torch.atan(1/sc)
        sc = self.arctan_sc(edge_attr_nei, edge_attr_support, dim=(-3, -2, -1))
        return sc

    def calculate_total_score(self, x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor):
        # calibrate p_neighbor
        p_neighbor = p_neighbor - p_focal.unsqueeze(1)

        # get kernel params
        x_center = self.x_center
        x_support = self.x_support
        edge_attr_support = self.edge_attr_support
        p_support = self.p_support

#         print('=====cal total sc')
#         print(f'x_center:{x_center.shape}')
#         print(f'x_support:{x_support.shape}')
#         print(f'edge_attr_support:{edge_attr_support.shape}')
#         print(f'p_support:{p_support.shape}')
#         print('\n')
#         print(f'x_focal:{x_focal.shape}')
#         print(f'p_focal:{p_focal.shape}')
#         print(f'x_neighbor:{x_neighbor.shape}')
#         print(f'p_neighbor:{p_neighbor.shape}')
#         print(f'edge_attr_neighbor:{edge_attr_neighbor.shape}')

        # calculate the angle score
        permuted_p_support: 'shape(num_permute, num_support, D)' = self.permute(
            p_support)

        angle_sc = self.get_angle_score(p_neighbor, permuted_p_support)
        best_angle_sc, best_angle_sc_index = torch.max(angle_sc, dim=1)
        best_angle_sc_index = best_angle_sc_index.unsqueeze(1)

#         print(f'angle_sc:{angle_sc.shape}')
#         print(angle_sc)
#         print(f'best_angle_sc:{best_angle_sc.shape}')
#         print(best_angle_sc.shape)
#         print('best_angle_sc_index:')
#         print(best_angle_sc_index.shape)

#         print(angle_sc)

        # get the length score. Firslty get the p and p_support combination that gives the best angle score
        selected_index = best_angle_sc_index.unsqueeze(-1).unsqueeze(-1).expand(
            best_angle_sc_index.shape[0], best_angle_sc_index.shape[1], permuted_p_support.shape[-2], permuted_p_support.shape[-1])
        best_p_support = torch.gather(permuted_p_support, 1, selected_index)
        # calculate the length score for the best combination
#         print(f'best_p_support:{best_p_support.shape}')
        length_sc = self.get_length_score(p_neighbor, best_p_support)
#         print(f'length_sc:{length_sc.shape}')

        # calculate the support attribute score for the best combination
        selected_index = best_angle_sc_index.unsqueeze(-1).unsqueeze(-1).expand(
            best_angle_sc_index.shape[0], best_angle_sc_index.shape[1], x_support.shape[-2], x_support.shape[-1])
        permuted_x_support = self.permute(x_support)
#         print(f'permuted_x_support:{permuted_x_support.shape}')
#         print(f'best_angle_sc_index:{best_angle_sc_index.shape}')
        best_x_support = torch.gather(permuted_x_support, 1, selected_index)
#         print(f'best_x_support:{best_x_support.shape}')
#         print(f'best_x_support:{best_x_support.shape}')
        supp_attr_sc = self.get_support_attribute_score(
            x_neighbor, best_x_support)
#         print('supp_attr_sc')
#         print(f'supp_attr_sc:{supp_attr_sc.shape}')

        # calculate the center attribute score
#         print(f'x_center:{x_center.shape}')
        center_attr_sc = self.get_center_attribute_score(x_focal, x_center)
#         print(f'center_attr_sc:{center_attr_sc.shape}')

        # calculate the edge attribute score
        selected_index = best_angle_sc_index.unsqueeze(-1).unsqueeze(-1).expand(
            best_angle_sc_index.shape[0], best_angle_sc_index.shape[1], edge_attr_support.shape[-2], edge_attr_support.shape[-1])
#         print(f'edge_attr_support:{edge_attr_support.shape}')
        permuted_edge_attr_support = self.permute(edge_attr_support)
#         print(f'permuted:{permuted_edge_attr_support.shape}')
#         print(f'best_angle_sc_index:{best_angle_sc_index.shape}')
        best_edge_attr_support = torch.gather(
            permuted_edge_attr_support, 1, selected_index)
#         print(f'best_edge_attr_support:{selected_index.shape}')
#         print(f'edge_attr_neighbor:{edge_attr_neighbor.shape}')
#         print(f'best_edge_attr_support:{best_edge_attr_support.shape}')
        edge_attr_support_sc = self.get_edge_attribute_score(
            edge_attr_neighbor, best_edge_attr_support)
#         print(f'edge_attr_support_sc:{edge_attr_support_sc.shape}')

        # convert each score to correct dimension
        angle_sc = best_angle_sc.unsqueeze(dim=0)
        length_sc = length_sc.unsqueeze(dim=0)
        supp_attr_sc = supp_attr_sc.unsqueeze(dim=0)
        center_attr_sc = center_attr_sc.unsqueeze(dim=0)
        edge_attr_support_sc = edge_attr_support_sc.unsqueeze(dim=0)

        # the maxium value a arctain function can get
        max_atan = torch.tensor([math.pi / 2])
        one = torch.tensor([1])

        sc = torch.atan(1 /
                        (torch.square(length_sc - max_atan) +
                         torch.square(angle_sc - max_atan) +
                         torch.square(supp_attr_sc - max_atan) +
                         torch.square(center_attr_sc - max_atan) +
                         torch.square(edge_attr_support_sc - max_atan)
                         )).squeeze(0)
#         print(f'cal total sc:{sc}')

        return sc, length_sc, angle_sc, supp_attr_sc, center_attr_sc, edge_attr_support_sc

    def forward(self, *argv, **kwargv):
        if len(kwargv) == 1:
            x_focal = kwargv['data'].x_focal
            p_focal = kwargv['data'].p_focal
            x_neighbor = kwargv['data'].x_neighbor
            p_neighbor = kwargv['data'].p_neighbor
            edge_attr_neighbor = kwargv['data'].edge_attr_neighbor
        else:
            x_focal = kwargv['x_focal']
            p_focal = kwargv['p_focal']
            x_neighbor = kwargv['x_neighbor']
            p_neighbor = kwargv['p_neighbor']
            edge_attr_neighbor = kwargv['edge_attr_neighbor']


#         x, x_focal, p, edge_attr, edge_index = self.convert_graph_to_receptive_field(x, p, edge_index, edge_attr)

        sc, length_sc, angle_sc, supp_attr_sc, center_attr_sc, edge_attr_support_sc = self.calculate_total_score(
            x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor)


#         print('\n')
#         print(f'len sc:{length_sc}')
#         print(f'angle sc:{angle_sc}')
#         print(f'support attribute_sc:{supp_attr_sc}')
#         print(f'edge attribute score:{edge_attr_support_sc}')
#         print(f'total sc: {sc.shape}')
        return sc  # , length_sc, angle_sc, supp_attr_sc, center_attr_sc, edge_attr_support_sc


class KernelSetConv(Module):
    def __init__(self, L, D, node_attr_dim, edge_attr_dim):
        super(KernelSetConv, self).__init__()
        self.L = L
        kernel1 = KernelConv(L=L, D=D, num_supports=1,
                             node_attr_dim=node_attr_dim, edge_attr_dim=edge_attr_dim)
        kernel2 = KernelConv(L=L, D=D, num_supports=2,
                             node_attr_dim=node_attr_dim, edge_attr_dim=edge_attr_dim)
        kernel3 = KernelConv(L=L, D=D, num_supports=3,
                             node_attr_dim=node_attr_dim, edge_attr_dim=edge_attr_dim)
        kernel4 = KernelConv(L=L, D=D, num_supports=4,
                             node_attr_dim=node_attr_dim, edge_attr_dim=edge_attr_dim)

        self.kernel_set = ModuleList([kernel1, kernel2, kernel3, kernel4])

#         kernel_set = ModuleList(
#             [KernelConv(D=D, num_supports=1, node_attr_dim = node_attr_dim, edge_attr_dim = edge_attr_dim),
#              KernelConv(D=D, num_supports=2, node_attr_dim = node_attr_dim, edge_attr_dim = edge_attr_dim),
#              KernelConv(D=D, num_supports=3, node_attr_dim = node_attr_dim, edge_attr_dim = edge_attr_dim),
#              KernelConv(D=D, num_supports=4, node_attr_dim = node_attr_dim, edge_attr_dim = edge_attr_dim)
#             ])

    def get_degree_index(self, x, edge_index):
        deg = degree(edge_index[0], x.shape[0])
        return deg

    def get_neighbor_index(self, edge_index, center_index):
        #         print('edge_index')
        #         print(edge_index)
        #         print('\n')
        #         print('center_index')
        #         print(center_index)
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = b.nonzero()
        d = c[:, 0]
        return edge_index[1, d]

    def get_focal_nodes_of_degree(self, deg, x, p, edge_index):
        '''
        outputs
        ori_x: a feature matrix that only contains rows (i.e. the center node) having certain degree
        ori_p: a position matrix that only contains rows (i.e. the center node) having certain degree
        '''
        deg_index = self.get_degree_index(x, edge_index)
        selected_index = (deg_index == deg).nonzero(as_tuple=True)
        x_focal = torch.index_select(input=x, dim=0, index=selected_index[0])
        p_focal = torch.index_select(input=p, dim=0, index=selected_index[0])

        return x_focal, p_focal

    def get_edge_attr_support_from_center_node(self, edge_attr, edge_index, center_index):
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = b.nonzero()
        d = c[:, 0]

        # normalize bond id
        e = (d / 2).long()
#         bond_id = torch.cat([torch.stack((2*x, 2*x+1)) for x in e])
        bond_id = torch.tensor([2 * x for x in e])
#         print('bond_id')
#         print(bond_id)

        # select bond attributes with the bond id
        nei_edge_attr = torch.index_select(
            input=edge_attr, dim=0, index=bond_id)

        return nei_edge_attr

    def get_neighbor_nodes_and_edges_of_degree(self, deg, x, p, edge_index, edge_attr):
        '''
        inputs:
        deg: the query degree
        num_focal: the number of focal nodes of degree deg in the graph

        outputs:
        nei_x: a feature matrix that only contains rows (i.e. the neighboring node) that its center node has certain degree
        nei_p: a position matrix that only contains rows (i.e. the neighboring node) that its center node has certain degree
        '''
        deg_index = self.get_degree_index(x, edge_index)
        center_index = (deg_index == deg).nonzero(as_tuple=True)[0]
        num_focal = len(center_index)
#         print('center_index')
#         print(center_index)

        nei_x_list = []
        nei_p_list = []
        nei_edge_attr_list = []
        for i in range(num_focal):
            nei_index = self.get_neighbor_index(edge_index, center_index[i])
#             print(f'nei_index:{nei_index.shape}')

            nei_x = torch.index_select(x, 0, nei_index)
#             print(f'nei_x:{nei_x.shape}')
            nei_p = torch.index_select(p, 0, nei_index)
#             print(f'nei_p:{nei_p.shape}')
            nei_edge_attr = self.get_edge_attr_support_from_center_node(
                edge_attr, edge_index, center_index[i])
#             print('\n nei_edge_attr')
#             print(nei_edge_attr)

            nei_x_list.append(nei_x)
            nei_p_list.append(nei_p)
            nei_edge_attr_list.append(nei_edge_attr)

        nei_x = torch.stack(nei_x_list, dim=0)
        nei_p = torch.stack(nei_p_list, dim=0)
        nei_edge_attr: 'shape(num_focal, num_support*2, num_support_attr_dim)' = torch.stack(
            nei_edge_attr_list, dim=0)

#         print('nei_edge_attr')
#         print(nei_edge_attr.shape)

        return nei_x, nei_p, nei_edge_attr

    def convert_graph_to_receptive_field(self, deg, x, p, edge_index, edge_attr):
        x_focal, p_focal = self.get_focal_nodes_of_degree(
            deg=deg, x=x, p=p, edge_index=edge_index)

        num_focal = x_focal.shape[0]
        print(f'num_focal:{num_focal}')
        if num_focal != 0:
            x_neighbor, p_neighbor, edge_attr_neighbor = self.get_neighbor_nodes_and_edges_of_degree(
                deg=deg, x=x, edge_index=edge_index, p=p, edge_attr=edge_attr)
#             print(f'x_neighbor:{x_neighbor.shape}')
#             print(f'p_neighbor:{p_neighbor.shape}')
            return x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor
        return None

    def forward(self, *argv, **kwargv):
        '''
        inputs:
        data: graph data containing feature matrix, adjacency matrix, edge_attr matrix
        '''
        if len(argv) != 0:
            raise Exception(
                'Kernel does not take positional argument, use keyword argument instead. e.g. model(data=data)')

        if len(kwargv) == 1:
            x = kwargv['data'].x
            edge_index = kwargv['data'].edge_index
            edge_attr = kwargv['data'].edge_attr
            p = kwargv['data'].p

        else:
            x = kwargv['x']
            edge_index = kwargv['edge_index']
            edge_attr = kwargv['edge_attr']
            p = kwargv['p']

#         print('edge_index')
#         print(edge_index)

#         print('edge_attr')
#         print(edge_attr)

        # loop through all possbile degrees. i.e. 1 to 4 bonds
        sc_list = []
        for deg in range(1, 5):
            #             print(f'deg:{deg}')
            receptive_field = self.convert_graph_to_receptive_field(
                deg, x, p, edge_index, edge_attr)
#             print('receptive_field')
#             print(receptive_field)
            if receptive_field is not None:
                x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor = receptive_field[
                    0], receptive_field[1], receptive_field[2], receptive_field[3], receptive_field[4]
                data = Data(x_focal=x_focal, p_focal=p_focal, x_neighbor=x_neighbor,
                            p_neighbor=p_neighbor, edge_attr_neighbor=edge_attr_neighbor)

#                 print('====data info====')
#                 print('x_focal')
#                 print(x_focal.shape)
#             print('p_focal')
#             print(p_focal)
#             print('x_neighbor')
#             print(x_neighbor)
#             print('p_neighbor')
#             print(p_neighbor)
#             print('edge_attr_neighbor')
#             print(edge_attr_neighbor)
                sc = self.kernel_set[deg - 1](data=data)
#                 print(f'sc.shape:{sc.shape}')

            else:
                sc = torch.tensor([0] * self.L)
                # the maxium value a arctain function can get
                max_atan = torch.tensor([math.pi / 2] * self.L)
                sc = max_atan

            sc_list.append(sc)

        sc_list = torch.stack(sc_list)
#         print(f'sc_list:{sc_list}')
        return sc_list


class KernelLayer(Module):
    def __init__(self, x_dim, p_dim, edge_dim, out_dim):
        '''
        a wrapper of KernelSetConv for clear input/output dimension
        inputs:
        D: dimension
        L: number of KernelConvSet
        '''
        super(KernelLayer, self).__init__()
        self.conv = KernelSetConv(
            L=out_dim, D=p_dim, node_attr_dim=x_dim, edge_attr_dim=edge_dim)

    def forward(self, data):
        return self.conv(data=data)
