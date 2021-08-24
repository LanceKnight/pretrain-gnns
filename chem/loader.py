import os
import torch
# import pickle
# import collections
# import math
import pandas as pd
import numpy as np
# import networkx as nx
from rdkit import Chem
# from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
# from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree
# from torch_geometric.data import Batch
# from itertools import repeat, product, chain
from tqdm import tqdm

# this pattern_dict is used for cleaning some smiles that cannot be processed by rdkit
pattern_dict = {'[NH-]': '[N-]'}

def generate_element_rep_list(elements):

    print('calculating rdkit element representation lookup table')
    elem_rep_lookup = []
    for elem in elements:
        pt = Chem.GetPeriodicTable()

        if isinstance(elem, int):
            num = elem
            sym = pt.GetElementSymbol(num)
        else:
            num = pt.GetAtomicNumber(elem)
            sym = elem
        w = pt.GetAtomicWeight(elem)

        Rvdw = pt.GetRvdw(elem)
    #     Rcoval = pt.GetRCovalent(elem)
        valence = pt.GetDefaultValence(elem)
        outer_elec = pt.GetNOuterElecs(elem)

        elem_rep = [num, w, Rvdw, valence, outer_elec]
#             print(elem_rep)

        elem_rep_lookup.append(elem_rep)
    elem_lst = elem_rep_lookup.copy()
    return elem_rep_lookup


max_elem_num = 118
element_nums = [x + 1 for x in range(max_elem_num)]
elem_lst = generate_element_rep_list(element_nums)


def get_atom_rep(atomic_num):
    '''use rdkit to generate atom representation
    '''
    global elem_lst

    result = 0
    try:
        result = elem_lst[atomic_num - 1]
    except:
        print(f'error: atomic_num {atomic_num} does not exist')

    return result


# def get_atom_rep(atomic_num, package='rdkit'):
#     '''use rdkit or pymatgen to generate atom representation
#     '''
#     global elem_lst

#     if package == 'rdkit':
#         elem_lst = lookup_from_rdkit(element_nums)
#     elif package == 'pymatgen':
#         raise Exception('pymatgen implementation is deprecated.')
#         # elem_lst = lookup_from_pymatgen(element_nums)
#     else:
#         raise Exception('cannot generate atom representation lookup table')

#     result = 0
#     try:
#         result = elem_lst[atomic_num - 1]
#     except:
#         print(f'error: atomic_num {atomic_num} does not exist')

#     return result


def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph generation.')
    # print(f'smiles:{smiles}')
    # default RDKit behavior is to reject hypervalent P, so you need to set sanitize=False. Search keyword = 'Explicit Valence Error - Partial Sanitization' on https://www.rdkit.org/docs/Cookbook.html for more info
    smiles = smiles.replace(r'/=', '=')
    smiles = smiles.replace(r'\=', '=')
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as e:
        print(f'Cannot generate mol, error:{e}, smiles:{smiles}')

    if mol is None:
        smiles = smiles_cleaner(smiles)
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except Exception as e:
            print(f'Generated mol is None, error:{e}, smiles:{smiles}')
            return None
    try:
        # mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f'{e}, smiles:{smiles}')

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    atom_pos = []
    atom_attr = []

    # get atom attributes and positions
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        h = get_atom_rep(atomic_num)

        if D == 2:
            atom_pos.append([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        elif D == 3:
            atom_pos.append([conf.GetAtomPosition(
                i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
        atom_attr.append(h)

    # get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, edge in enumerate(mol.GetBonds()):
        i = edge.GetBeginAtomIdx()
        j = edge.GetEndAtomIdx()

        bond_attr = None
        bond_type = edge.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_attr = [1]
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_attr = [2]
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_attr = [3]
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_attr = [4]

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)
#         print(f'i:{i} j:{j} bond_attr:{bond_attr}')

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)
#         print(f'j:{j} j:{i} bond_attr:{bond_attr}')

    x = torch.tensor(atom_attr)
    p = torch.tensor(atom_pos)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list)

    data = Data(x=x, p=p, edge_index=edge_index, edge_attr=edge_attr)
    return data


# def mol2graph(mol):


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 D=2,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):

        self.dataset = dataset
        self.root = root
        self.D = D
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    # def get(self, idx):
    #     data = Data()
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         s = list(repeat(slice(None), item.dim()))
    #         s[data.__cat_dim__(key, item)] = slice(slices[idx],
    #                                                 slices[idx + 1])
    #         data[key] = item[s]
    #     return data

    @ property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @ property
    def processed_file_names(self):
        return f'{self.dataset}-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    # def __getitem__(self, index):
    #     return self.get(index)

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset not in ['435008', '1798', '435034']:
            raise ValueError('Invalid dataset name')

        for file, label in [(f'{self.dataset}_actives.smi', 1),
                            (f'{self.dataset}_inactives.smi', 0)]:
            smiles_path = os.path.join(self.root, 'raw', file)
            smiles_list = pd.read_csv(
                smiles_path, sep='\t', header=None)[0]

            for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
            # for i in tqdm(range(1)):
                smi = smiles_list[i]

                data = smiles2graph(2, smi)
                if data is None:
                    continue
                data.id = torch.tensor([i])
                data.y = torch.tensor([label])
                data.smiles = smi
                # print(data)
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print('doing pre_transforming...')
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(
            self.processed_dir, f'{self.dataset}-smiles.csv'), index=False, header=False)

        for data in data_list:
            print(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain


# class SDFBenchmakr2015(InMemoryDataset):
#     def __init__(self,
#                  root,
#                  D=3,
#                  transform=None,
#                  pre_transform=None,
#                  pre_filter=None,
#                  dataset='435008',
#                  empty=False):

#         self.dataset = dataset
#         self.root = root
#         self.D = D
#         super(SDFBenchmakr2015, self).__init__(root, transform, pre_transform,
#                                                pre_filter)
#         self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

#         if not empty:
#             self.data, self.slices = torch.load(self.processed_paths[0])

#     @ property
#     def raw_file_names(self):
#         file_name_list = os.listdir(self.raw_dir)
#         # assert len(file_name_list) == 1     # currently assume we have a
#         # # single raw file
#         return file_name_list

#     @ property
#     def processed_file_names(self):
#         return f'geometric_data_processed-{self.D}D.pt'

#     def download(self):
#         raise NotImplementedError('Must indicate valid location of raw data. '
#                                   'No download allowed')

#     def process(self):
#         data_smiles_list = []
#         data_list = []

#         if self.dataset == '435008':
#             for file, label in [(f'{self.dataset}_actives_clean.smi', 1),
#                                 (f'{self.dataset}_inactives_clean.smi', 0)]:
#                 smiles_path = os.path.join(self.root, 'raw', file)
#                 smiles_list = pd.read_csv(smiles_path, sep='\t', header=None)[0]

#                 for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
#                     smi = smiles_list[i]

#                     data = smiles2graph(2, smi)
#                     if data is None:
#                         continue
#                     data.id = torch.tensor([i])
#                     data.y = torch.tensor([label])
#                     data.smiles = smi
#                     # print(data)
#                     data_list.append(data)
#                     data_smiles_list.append(smiles_list[i])

#         else:
#             raise ValueError('Invalid dataset name')

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         # write data_smiles_list in processed paths
#         data_smiles_series = pd.Series(data_smiles_list)
#         data_smiles_series.to_csv(os.path.join(self.processed_dir,
#                                                'smiles.csv'), index=False,
#                                   header=False)

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


# def create_circular_fingerprint(mol, radius, size, chirality):
#     """

#     :param mol:
#     :param radius:
#     :param size:
#     :param chirality:
#     :return: np array of morgan fingerprint
#     """
#     fp = GetMorganFingerprintAsBitVect(mol, radius,
#                                        nBits=size, useChirality=chirality)
#     return np.array(fp)


def smiles_cleaner(smiles):
    '''
    this function is to clean smiles for some known issues that makes rdkit:Chem.MolFromSmiles not working
    '''
    print('fixing smiles for rdkit...')
    new_smiles = smiles
    for pattern, replace_value in pattern_dict.items():
        if pattern in smiles:
            print('found pattern and fixed the smiles!')
            new_smiles = smiles.replace(pattern, replace_value)
    return new_smiles


class ToIndexAndEdgeAttrForDeg(object):
    '''
    Calculate the focal index and neighbor indices for each degree and store them in focal_index and nei_index.
    Also calculate neighboring edge attr for each degree nei_edge_attr.
    These operations are very expensive to run on the fly
    '''

    def get_neighbor_index(self, edge_index, center_index):
        #         print('edge_index')
        #         print(edge_index)
        #         print('\n')
        #         print('center_index')
        #         print(center_index)
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = torch.nonzero(b)
        d = c[:, 0]
        return edge_index[1, d]

    def get_degree_index(self, x, edge_index):
        # print(f'edge_index:{edge_index.shape}, x:{x.shape}')
        deg = degree(edge_index[0], x.shape[0])
        return deg

    def get_edge_attr_support_from_center_node(self, edge_attr, edge_index, center_index):
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = torch.nonzero(b)
        d = c[:, 0]

        # normalize bond id
        e = (d / 2).long()
#         bond_id = torch.cat([torch.stack((2*x, 2*x+1)) for x in e])
        bond_id = torch.tensor([2 * x for x in e], device=a.device)
#         print('bond_id')
#         print(bond_id)

        # select bond attributes with the bond id
        nei_edge_attr = torch.index_select(input=edge_attr, dim=0, index=bond_id)

        return nei_edge_attr

    def __call__(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        deg_index = self.get_degree_index(x, edge_index)

        deg = 1
        data.focal_index_deg1 = (deg_index == deg).nonzero(as_tuple=True)[0]
        num_focal = len(data.focal_index_deg1)
        nei_index_deg1_list = []
        nei_edge_attr_deg1_list = []
        for i in range(num_focal):
            nei_index_deg1_list.append(self.get_neighbor_index(edge_index, data.focal_index_deg1[i]))
            nei_edge_attr_deg1_list.append(self.get_edge_attr_support_from_center_node(edge_attr, edge_index, data.focal_index_deg1[i]))
        if len(nei_index_deg1_list) != 0:
            data.nei_index_deg1 = torch.stack(nei_index_deg1_list, dim=0).T
            data.nei_edge_attr_deg1 = torch.stack(nei_edge_attr_deg1_list, dim=0)
        else:
            data.nei_index_deg1 = torch.Tensor()
            data.nei_edge_attr_deg1 = torch.Tensor()

        deg = 2
        data.focal_index_deg2 = (deg_index == deg).nonzero(as_tuple=True)[0]
        num_focal = len(data.focal_index_deg2)
        nei_index_deg2_list = []
        nei_edge_attr_deg2_list = []
        for i in range(num_focal):
            nei_index_deg2_list.append(self.get_neighbor_index(edge_index, data.focal_index_deg2[i]))
            nei_edge_attr_deg2_list.append(self.get_edge_attr_support_from_center_node(edge_attr, edge_index, data.focal_index_deg2[i]))
        if len(nei_index_deg2_list) != 0:
            data.nei_index_deg2 = torch.stack(nei_index_deg2_list, dim=0).T
            data.nei_edge_attr_deg2 = torch.stack(nei_edge_attr_deg2_list, dim=0)
        else:
            data.nei_index_deg2 = torch.Tensor()
            data.nei_edge_attr_deg2 = torch.Tensor()

        deg = 3
        data.focal_index_deg3 = (deg_index == deg).nonzero(as_tuple=True)[0]
        num_focal = len(data.focal_index_deg3)
        nei_index_deg3_list = []
        nei_edge_attr_deg3_list = []
        for i in range(num_focal):
            nei_index_deg3_list.append(self.get_neighbor_index(edge_index, data.focal_index_deg3[i]))
            nei_edge_attr_deg3_list.append(self.get_edge_attr_support_from_center_node(edge_attr, edge_index, data.focal_index_deg3[i]))
        if len(nei_index_deg3_list) != 0:
            data.nei_index_deg3 = torch.stack(nei_index_deg3_list, dim=0).T
            data.nei_edge_attr_deg3 = torch.stack(nei_edge_attr_deg3_list, dim=0)
        else:
            data.nei_index_deg3 = torch.Tensor()
            data.nei_edge_attr_deg3 = torch.Tensor()

        deg = 4
        data.focal_index_deg4 = (deg_index == deg).nonzero(as_tuple=True)[0]
        num_focal = len(data.focal_index_deg4)
        nei_index_deg4_list = []
        nei_edge_attr_deg4_list = []
        for i in range(num_focal):
            nei_index_deg4_list.append(self.get_neighbor_index(edge_index, data.focal_index_deg4[i]))
            nei_edge_attr_deg4_list.append(self.get_edge_attr_support_from_center_node(edge_attr, edge_index, data.focal_index_deg4[i]))
        if len(nei_index_deg4_list) != 0:
            data.nei_index_deg4 = torch.stack(nei_index_deg4_list, dim=0).T
            data.nei_edge_attr_deg4 = torch.stack(nei_edge_attr_deg4_list, dim=0)
        else:
            data.nei_index_deg4 = torch.Tensor()
            data.nei_edge_attr_deg4 = torch.Tensor()

        return data

        # test MoleculeDataset object
if __name__ == "__main__":
    print('testing...')
    dataset = '435034'
    # windows
    # root = 'D:/Documents/JupyterNotebook/GCN_property/pretrain-gnns/chem/dataset/'
    # linux
    root = 'dataset/'
    if dataset == '435008' or '1798' or '435034':
        root += 'qsar_benchmark2015'
        dataset = dataset
    else:
        raise Exception('cannot find dataset')
    dataset = MoleculeDataset(D=2, root=root, dataset=dataset, pre_transform=ToIndexAndEdgeAttrForDeg())
    print(dataset[0])

    data = dataset[0]
    print(data.x)
    # print(data.p)
    # print(data.edge_index)
    # print(data.edge_attr)
    # create_all_datasets()
