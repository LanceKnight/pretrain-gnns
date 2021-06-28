import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn import CosineSimilarity as cos
from torch.nn import Sigmoid
from itertools import permutations


from loader import mol_to_graph_data


# class Position():
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z

#     def __repr__(self):
#         return f'({self.x}, {self.y}, {self.z})'


class Node():
    def __init__(self, position, feature):
        self._position = position
        self._feature = feature

    def __repr__(self):
        return f'pos:{self.feature}, prop:{self.feature}'

    @property
    def f(self):
        return self._feature

    @h.setter
    def f(self, feature):
        self._feature = feature


class Kernel():
    def __init__(self, num_support):
        f = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        pos = torch.tensor([1, 2, 3], dtype=torch.float)
        self.center = Node(pos, f)
        self.num_support = num_support

        self.support_list = []
        for i in range(self.num_support):
            node = Node(pos, f)
            self.support_list.append(node)

    def __repr__(self):
        return f'center:{str(self.center)}, supports:{self.support_list}'

    def __int__(self):
        return self.num_support


class Kernel_1(Kernel):
    def __init__(self):
        super(Kernel_1, self).__init__(1)


class Kernel_2(Kernel):
    def __init__(self):
        super(Kernel_2, self).__init__(2)


class Kernel_3(Kernel):
    def __init__(self):
        super(Kernel_3, self).__init__(3)


class Kernel_4(Kernel):
    def __init__(self):
        super(Kernel_4, self).__init__(4)


def sim(patch_neighbor, kernel_support, patch_focal):
    dot_product = torch.mm(patch_neighbor.f, kernel_support.f)
    cos_sim = cos()(patch_neighbor.position - patch_focal.postion, kernel_support.position)
    return dot_product * cos_sim


def permute(x):
    '''
    return a list of tensors that have all row permuations
    '''
    rows = x.shape[0]
    l = [x[torch.tensor(index_order)]
         for index_order in list(permutations(range(rows)))]
    return l


def get_rows_of_degree_k(X, P, edge_index, num_neighbors):
    '''
    return a tuple of feature tensor, position tensor and index of selected rows
    rows are whose corresponding nodes have the required number of neighbors k
    '''
    deg = degree(edge_index[0], X.shape[0])
    selected_index = (deg == num_neighbors).nonzero(as_tuple=True)
    X_k = torch.index_select(X, 0, index[0])
    P_k = torch.index_select(P, 0, index[0])
    return X_k, P_k, selected_index


def sim(X, P, X_s, P_s, selected_index, edge_index):
    '''
    calculate the sim term in the paper
    sigmoid(<f_m⋅f_s>*CosineSimilarity(p_m – p_n, p_s))
    '''

    term1 = sigmoid(torch.mm(X, P))  # i.e. <f_m⋅f_s>
    cos = CosineSimilarity(dim=1)

    activation = Sigmoid()
    term1 = activation()(term1)


def graph_conv(X, P, edge_index, X_s, P_s, X_c, edge_index_k):
    '''
    X, P, edge_index are feature matrix, position matrix and connectivity matrix of the graph, respectively
    X_s, P_s are feature matrix, position matrix of the kernel supports, respectively
    X_c is the feature of kernel center
    edge_index_k is the connectivity of the kernel

    formula using symbols from the paper
    graph_conv = sigmoid(<f_n, f_c>) + max{}
    '''

    term1 = torch.mm(X, X_c.T)  # i.e. <f_n, f_c>
    activation = Sigmoid()
    term1 = activation()(term1)

    # only keep the nodes that have the same number of neighbors as the number of kernel supports
    number_neighbors = X_s.shape[0]
    X, P, selected_index = get_rows_of_degree_k(
        X, P, edge_index, number_neighbors)


class GraphConv3D(MessagePassing):
    def __init__(self):
        super(GraphConv3D, self).__init__()
        self.kernel = None
        self.kernel_1 = Kernel_1()
        self.kernel_2 = Kernel_2()
        self.kernel_3 = Kernel_3()
        self.kernel_4 = Kernel_4()
        self.kernel_list = [[Kernel_1()], [Kernel_2()], [
            Kernel_3()], [Kernel_4()]]

    def forward(self, *argv):
        if len(argv) == 1:
            x = argv[0].x
            edge_index = argv[0].edge_index
            edge_attr = argv[0].edge_attr
        elif len(argv) == 3:
            x = argv[0]
            edge_index = argv[1]
            edge_attr = argv[2]
        else:
            raise Exception('Wrong number of inputs for GraphConv3D')

        # print(f'deg:{self.deg}')
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # kernel_conv()
        pass
        # for x in x_i:


if __name__ == '__main__':
    smiles = 'CC(C)Nc1nc(NC(C)C)nc(n1)n2nc(C)cc2C'
    smiles = 'C(Br)(I)(OF)Cl'
    mol = Chem.MolFromSmiles(smiles)

    data = mol_to_graph_data(mol)
    print(data)
    print(data.deg)
    print(data.x)

    model = GraphConv3D()
    # output = model(data)

    k = Kernel_1()
    print(k.center.f)
