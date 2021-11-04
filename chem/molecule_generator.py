import rdkit
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
from random import randint
import networkx as nx
from wrapper import smiles2graph


def randomly_add_node(data):
    old_graph = to_networkx(data)
    old_nodes = old_graph.nodes
    num_old_nodes = len(old_nodes)
    randn = randint(0, num_old_nodes - 1)

    new_node = torch.tensor([-1])
    x = torch.cat((data.x, new_node.unsqueeze(-1)), dim=0)

    new_edge = torch.tensor([[randn, num_old_nodes], [num_old_nodes, randn]])
    edge_index = torch.cat((data.edge_index, new_edge), dim=1)
    print(edge_index)
    data = Data(x=x, edge_index=edge_index)
    return data


def generate_2D_molecule_from_reference(smiles, num):
    '''generate molecules with similar connectivity with the reference molecule
    smiles: input molecule
    num: number of augmented molecules to generate
    '''
