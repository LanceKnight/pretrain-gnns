import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
# from torch_geometric.utils import add_self_loops
from torch.utils.data.dataset import random_split
from torch.nn import Module, Linear, Embedding
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from clearml import Task
from torch.nn import CosineSimilarity
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import os
from monitors import LossMonitor
import torch
from torch import Tensor
from dataset import QSARDataset
from torch.utils.data import WeightedRandomSampler
from gin_conv import GINEConv
from model import GNN_graphpred

# from wrapper import smiles2graph

max_epochs = 10
generate_num = 100
batch_size = 16


# def maybe_num_nodes(edge_index, num_nodes=None):
#     if num_nodes is not None:
#         return num_nodes
#     elif isinstance(edge_index, Tensor):
#         return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
#     else:
#         return max(edge_index.size(0), edge_index.size(1))


# def add_self_loops(
#         edge_index, edge_attr=None,
#         fill_value=None,
#         num_nodes=None):
#     r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
#     :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
#     In case the graph is weighted or has multi-dimensional edge features
#     (:obj:`edge_attr != None`), edge features of self-loops will be added
#     according to :obj:`fill_value`.

#     Args:
#         edge_index (LongTensor): The edge indices.
#         edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
#             features. (default: :obj:`None`)
#         fill_value (float or Tensor or str, optional): The way to generate
#             edge features of self-loops (in case :obj:`edge_attr != None`).
#             If given as :obj:`float` or :class:`torch.Tensor`, edge features of
#             self-loops will be directly given by :obj:`fill_value`.
#             If given as :obj:`str`, edge features of self-loops are computed by
#             aggregating all features of edges that point to the specific node,
#             according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
#             :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
#         num_nodes (int, optional): The number of nodes, *i.e.*
#             :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

#     :rtype: (:class:`LongTensor`, :class:`Tensor`)
#     """
#     N = maybe_num_nodes(edge_index, num_nodes)

#     loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
#     loop_index = loop_index.unsqueeze(0).repeat(2, 1)

#     if edge_attr is not None:
#         if fill_value is None:
#             loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

#         elif isinstance(fill_value, float) or isinstance(fill_value, int):
#             loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
#                                            fill_value)
#         elif isinstance(fill_value, Tensor):
#             loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
#             if edge_attr.dim() != loop_attr.dim():
#                 loop_attr = loop_attr.unsqueeze(0)
#             sizes = [N] + [1] * (loop_attr.dim() - 1)
#             loop_attr = loop_attr.repeat(*sizes)

#         elif isinstance(fill_value, str):
#             loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
#                                 reduce=fill_value)
#         else:
#             raise AttributeError("No valid 'fill_value' provided")

#         edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

#     edge_index = torch.cat([edge_index, loop_index], dim=1)
#     return edge_index, edge_attr


# class GINConv(MessagePassing):
#     """
#     Extension of GIN aggregation to incorporate edge information by concatenation.
#     Args:
#         emb_dim (int): dimensionality of embeddings for nodes and edges.
#         embed_input (bool): whether to embed input or not.

#     See https://arxiv.org/abs/1810.00826
#     """

#     def __init__(self, emb_dim, aggr="add"):
#         super(GINConv, self).__init__()
#         # multi-layer perceptron
#         self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
#         self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
#         self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

#         torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
#         torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
#         self.aggr = aggr

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

#         # add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x.size(0), 2)
#         self_loop_attr[:, 0] = 4  # bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

#         edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

#         return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

#     def message(self, x_j, edge_attr):
#         return x_j + edge_attr

#     def update(self, aggr_out):
#         return self.mlp(aggr_out)





class MyDataset(pl.LightningDataModule):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.batch_size = batch_size
        self.metric = 'loss'

    def setup(self):
        dataset = QSARDataset()
        self.train_set = dataset[[torch.tensor(x) for x in range(0, 326)] + [torch.tensor(x) for x in range(1000, 10674)]]
        self.val_set = dataset[[torch.tensor(x) for x in range(326, 362)] + [torch.tensor(x) for x in range(20000, 29964)]]

    def train_dataloader(self):
        num_train_active = len(torch.nonzero(torch.tensor([data.y for data in self.train_set])))
        num_train_inactive = len(self.train_set) - num_train_active
        print(f'training size: {len(self.train_set)}, actives: {num_train_active}')
        train_sampler_weight = torch.tensor([(1. / num_train_inactive) if data.y == 0 else (1. / num_train_active) for data in self.train_set])

        train_sampler = WeightedRandomSampler(train_sampler_weight, len(train_sampler_weight))

        loader = DataLoader(self.train_set, batch_size=self.batch_size, sampler=train_sampler)
        return loader

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    # def teardown(self):
    #     print('sth...')


# task = Task.init(project_name="Tests/QSAR-Baseline", task_name="running test", tags=["debug"])

def main():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='435034',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str,
                        default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold",
                        help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0,
                        help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for dataset loading')
    args = parser.parse_args()

    # logger = task.get_logger()
    print(f'starting')
    dm = MyDataset()
    dm.prepare_data()
    dm.setup()
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks=1, JK=args.JK,
                          drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    # trainer = Trainer(max_epochs=3)

    dirpath = 'records'
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename='checkpoint')

    parser = ArgumentParser()
    args = parser.parse_args()
    if os.path.exists(dirpath + '/checkpoint.ckpt'):
        args.resume_from_checkpoint = dirpath + '/checkpoint.ckpt'
    trainer = Trainer.from_argparse_args(args)
    trainer.max_epochs = max_epochs
    trainer.callbacks.append(checkpoint_callback)
    # trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='step'))
    # trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='epoch'))
    # data = QSARDataset()[0]
    # output = model(data)

    #=============================
    trainer.fit(model, dm)
    # trainer.test(model, dm)

    # print('================')
    emb_list = []
    for data in loader:
        # print(f'data after:{data.x}')
        emb_list.append(model(data))
        # print(f'emb after:{emb}')
    # for param in model.parameters():
    #     print(f'after{param}')
    dist1 = CosineSimilarity()(emb_list[0], emb_list[1])
    dist2 = CosineSimilarity()(emb_list[0], emb_list[2])
    print(f'neg dist:{dist1}')
    print(f'pos dist:{dist2}')
    print(f'done')


if __name__ == '__main__':
    main()
