from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import rdmolops
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from extra_utils import get_atom_rep

allowable_features = {
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]


}


def graph_from_mol(mol, **kwargv):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms

    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        h = get_atom_rep(atom.GetAtomicNum(), 'rdkit')
        # print(h)

        atom_feature = get_atom_rep(atom.GetAtomicNum(
        ), 'rdkit') + [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]

        # atom_feature = [allowable_features['possible_atomic_num_list'].index(
        #     atom.GetAtomicNum())] + [allowable_features[
        #     'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                    'possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    AllChem.EmbedMolecule(mol)
    try:
        volume = AllChem.ComputeMolVolume(mol)
    except:
        print(f'cannot get volume, smiles:{Chem.MolToSmiles(mol)}')
        return None

    if kwargv['smiles'] is not None:
        smiles = kwargv['smiles']
    else:
        smiles = Chem.MolToSmiles(mol)

    if kwargv['weight'] is not None:
        weight = kwargv['weight']
    else:
        weight = Chem.Descriptors.ExactMolWt(mol)
    weight = torch.tensor(weight)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                volume=volume, weight=weight, smiles=smiles)

    return data

# elements = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
# bonds = ['-', '=', '#']


# def is_valid(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#     except:
#         print(f'error:{smiles}')

#     if mol is not None:
#         # rdmolops.AddHs(mol)
#         try:
#             Chem.SanitizeMol(mol)
#             return True
#         except:
#             print(f'{smiles} is not chemically valid')
#             return False
#     return False


# def add_atom(input_smiles):
#     new_list = []
#     print(f'input:{input_smiles}')
#     for elem in elements:
#         for bond in bonds:
#             smiles = input_smiles
#             smiles += bond
#             smiles += elem

#             print(f'generated:{smiles}')
#             if is_valid(smiles):
#                 print(f'validated:{smiles}')
#                 new_list.append(smiles)

#     return new_list


# def generate_molecule_list(num_atoms=2):

#     molecule_lst = []
#     generation_seeds = elements
#     temp_seeds = []

#     # for num in range(num_atoms - 1):
#     for smiles in generation_seeds:

#         # print(f'before adding:{smiles}')

#         smiles_list = add_atom(smiles)
#         molecule_lst += smiles_list

#     generation_seeds = molecule_lst.copy()

#     print(f'generation_seeds:{generation_seeds}')
#     for smiles in generation_seeds:

#         #     # print(f'before adding:{smiles}')

#         smiles_list = add_atom(smiles)
#     molecule_lst += smiles_list

#     return molecule_lst

class weight_filter():
    def __call__(self, mol, std_w):
        w = Chem.Descriptors.ExactMolWt(mol)
        if w < self.std_w:
            return True
        else:
            return False


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 # cannot use range(0,20), it will generate a range between (0,1) instead of (0,20)
                 molecular_weight=(0, 20),
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        # do not process molecules outside this weight range
        self.w_range = molecular_weight

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
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

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return f'processed_pcba_{self.w_range}.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        file_path = os.path.join(self.root, 'raw', 'pcba.csv')
        smiles_list = pd.read_csv(file_path)['smiles']
        for smi in tqdm(smiles_list, desc='data processing progress'):
            if '.' in smi:
                continue

            mol = Chem.MolFromSmiles(smi)
            mol = AddHs(mol)
            w = ExactMolWt(mol)

            if self.w_range[0] < w < self.w_range[1]:
                # print(
                #     f'w:{w} range[0]:{self.w_range[0]}, range[0]:{self.w_range[1]},condition:{self.w_range[0] < w < self.w_range[1]}')
                data = graph_from_mol(mol, weight=w, smiles=smi)

                if data is not None:
                    data_list.append(data)

        if(len(data_list) > 0):
            print('database size:{len(data_list)}')
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            print('empty data')


if __name__ == '__main__':
    print('testing')

    for a in range(0, 1000, 50):

        w_range = (a, a + 50)
        print(f'range:{w_range}')
        dataset = MoleculeDataset(
            root='dataset/pcba', molecular_weight=w_range)
    # lst = generate_molecule_list(3)
    # print(f'data size:{len(lst)}')
    # for smi in lst:
    #     print(smi)
    #     if is_valid(smi) == False:
    #         print(f'{smi} is not valid')
