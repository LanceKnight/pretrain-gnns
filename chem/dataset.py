from torch_geometric.data import InMemoryDataset, Data
from ogb.utils import smiles2graph as ogb_smiles2graph
import torch
import os
from tqdm import tqdm
import pandas as pd


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocessing(item):
    item.x = convert_to_single_emb(item.x)
    return item


class QSARDataset(InMemoryDataset):
    def __init__(self,
                 root='dataset/qsar_benchmark2015/',
                 D=3,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='435034',
                 empty=False):

        self.dataset = dataset
        self.root = root
        self.D = D
        super(QSARDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @ property
    def processed_file_names(self):
        return f'{self.dataset}-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset not in ['435008', '1798', '435034']:
            # print(f'dataset:{self.dataset}')
            raise ValueError('Invalid dataset name')

        for file, label in [(f'{self.dataset}_actives.smi', 1),
                            (f'{self.dataset}_inactives.smi', 0)]:
            smiles_path = os.path.join(self.root, 'raw', file)
            smiles_list = pd.read_csv(
                smiles_path, sep='\t', header=None)[0]

            # only get first 100 data
            smiles_list = smiles_list

            for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
                # for i in tqdm(range(1)):
                smi = smiles_list[i]

                try:
                    graph_dict = ogb_smiles2graph(smi)
                except:
                    print('cannot convert smiles to graph')
                    pass

                data = Data()
                data.__num_nodes__ = int(graph_dict['num_nodes'])
                data.edge_index = torch.from_numpy(graph_dict['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph_dict['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph_dict['node_feat']).to(torch.int64)

                data.idx = i
                data.y = torch.tensor([label], dtype=torch.float32)
                data.smiles = smi

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

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = {}
        # total 362 actives. split: train-290, 36, 36
        split_dict['train'] = [torch.tensor(x) for x in range(0, 326)] + [torch.tensor(x) for x in range(1000, 10674)]  # training
        # split_dict['valid'] = [torch.tensor(x) for x in range(0, 290)] + [torch.tensor(x) for x in range(1000, 1510)] # 800 training
        split_dict['valid'] = [torch.tensor(x) for x in range(326, 362)] + [torch.tensor(x) for x in range(20000, 29964)]  # 100 valid
        split_dict['test'] = [torch.tensor(x) for x in range(326, 362)] + [torch.tensor(x) for x in range(3000, 9066)]  # 100 test
        # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocessing(item)
        else:
            return self.index_select(idx)


if __name__ == '__main__':
    dataset = QSARDataset()
    data = dataset[0]
    print(data.x)
