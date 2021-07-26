import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

from tqdm import tqdm
import numpy as np


from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil
import platform

from tensorboardX import SummaryWriter

from clearml import Task

from model import GNN_graphpred
from evaluation import enrichment, roc_auc, ppv
from util import print_model_size

criterion = nn.BCEWithLogitsLoss(reduction='sum')
# criterion = nn.BCELoss()


def train(args, model, device, loader, optimizer):
    model.train()

    loss_lst = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch.batch = batch.batch.to(device)

        batch = batch.to(device)

        pred, h = model(batch.x, batch.p, batch.edge_index,
                        batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # print('pred:')
        # print(pred)
        # print('y:')
        # print(y)

        loss = criterion(pred, y)
        # print(f'loss:{loss}')

        # for name, param in model.named_parameters():
        #     print(f'{name}: {param}')

        loss_lst.append(loss)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    batch_loss = sum(loss_lst) / float(len(loader))
    print(f'loss:{batch_loss}')
    return batch_loss


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred, h = model(batch.x, batch.p, batch.edge_index,
                            batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        print(f'batch.y:{batch.y}')

        print(f'pred:{pred}')
        pred = torch.where(pred > 0.5, torch.ones(
            pred.shape, device=device), torch.zeros(pred.shape, device=device))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    enrichment_list = []
    roc_auc_list = []
    ppv_list = []

    for i in range(y_true.shape[1]):
        enrichment_list.append(enrichment(y_true[:, i], y_scores[:, i]))
        roc_auc_list.append(roc_auc(y_true[:, i], y_scores[:, i]))
        ppv_list.append(ppv(y_true[:, i], y_scores[:, i]))
    # for i in range(y_true.shape[1]):
    #     # AUC is only defined when there is at least one positive data.
    #     if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
    #         is_valid = y_true[:, i]**2 > 0
    #         enrichment_list.append(roc_auc_score(
    #             (y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(enrichment_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(enrichment_list)) / y_true.shape[1]))

    return sum(enrichment_list) / len(enrichment_list), sum(roc_auc_list) / len(roc_auc_list), sum(ppv_list) / len(ppv_list)


def main():
    # start clearml as the task manager. this is optional.

    task = Task.init(project_name="kernel GNN", task_name="more layers")
    logger = task.get_logger()

    # ==========settings==========
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
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--num_kernel1', type=int, default=10, help='number of kernel of degree1 (default: 10).')
    parser.add_argument('--num_kernel2', type=int, default=10, help='number of kernel of degree2 (default: 10).')
    parser.add_argument('--num_kernel3', type=int, default=10, help='number of kernel of degree3 (default: 10).')
    parser.add_argument('--num_kernel4', type=int, default=10, help='number of kernel of degree4 (default: 10).')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='435008',
                        help='name of the dataset')
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
    parser.add_argument('--D', type=int, default=3,
                        help='dimension to work on, either 2D or 3D. (default:3)')
    args = parser.parse_args()

    # ==========seeds==========
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # ==========set up dataset==========
    # windows
    if platform.system() == 'Windows':
        root = 'D:/Documents/JupyterNotebook/GCN_property/pretrain-gnns/chem/dataset/'
    # linux
    else:
        root = '~/projects/GCN_Syn/examples/pretrain-gnns/chem/dataset/'
    if args.dataset == '435008':
        root = root + 'qsar_benchmark2015'
        dataset = args.dataset
    else:
        raise Exception('cannot find dataset')

    print(f'woring on {args.D}D now...')

    dataset = MoleculeDataset(D=args.D, root=root, dataset=dataset)
    print(f'dataset[0]:{dataset[0]}')
    index = list(range(1000))
    dataset = dataset[index]
    print(f'dataset:{dataset}')

    # ==========splitting datasets==========
    if args.split == "scaffold":
        smiles_list = pd.read_csv(
            "D:/Documents/JupyterNotebook/Hit_Explosion/data/lit-pcba/VAE/" + args.dataset.upper() + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(
            'dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print("random scaffold")
    elif args.split == 'lit-pcba':
        # train_dataset
        pass
    else:
        raise ValueError("Invalid split option.")

    # =======sampling dataset========
    num_train_active = len(torch.nonzero(torch.tensor([data.y for data in train_dataset])))
    num_train_inactive = len(train_dataset) - num_train_active
    print(f'training size: {len(train_dataset)}, actives: {num_train_active}')
    num_valid_active = len(torch.nonzero(torch.tensor([data.y for data in valid_dataset])))
    num_valid_inactive = len(valid_dataset) - num_valid_active
    print(f'valid size: {len(valid_dataset)}, actives: {num_valid_active}')
    num_test_active = len(torch.nonzero(torch.tensor([data.y for data in test_dataset])))
    num_test_inactive = len(test_dataset) - num_test_active
    print(f'test size: {len(test_dataset)}, actives: {num_test_active}')

    print('loading data')
    train_sampler_weight = torch.tensor([(1. / num_train_inactive) if data.y == 0 else (1. / num_train_active) for data in train_dataset])
    # valid_sampler_weight = torch.tensor([1. / num_valid_inactive, 1. / num_valid_active])
    # test_sampler_weight = torch.tensor([1. / num_test_inactive, 1. / num_test_active])

    # print(f'weights:valid total:{len(valid_sampler_weight)}, actives:{valid_sampler_weight.tolist().count(1. / num_valid_active)}, inactives:{valid_sampler_weight.tolist().count(1. / num_valid_inactive)}')
    # print(f'weights:test total:{len(test_sampler_weight)}, actives:{test_sampler_weight.tolist().count(1. / num_test_active)}, inactives:{test_sampler_weight.tolist().count(num_test_inactive)}')
    train_sampler = WeightedRandomSampler(train_sampler_weight, len(train_sampler_weight))
    # valid_sampler = WeightedRandomSampler(valid_sampler_weight, len(valid_dataset))
    # test_sampler = WeightedRandomSampler(test_sampler_weight, len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler)
    val_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, num_workers=args.num_workers)  # , sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=args.num_workers)  # , sampler=test_sampler)
    print('data loaded!')
    # for batch in train_loader:
    #     target = batch.y
    #     print(f"batch index , 0/1:{len(np.where(target.numpy()==0)[0])}/{len(np.where(target.numpy()==1)[0])}")

    # ==========set up model==========
    model = GNN_graphpred(num_layers=args.num_layers, num_kernel1=args.num_kernel1, num_kernel2=args.num_kernel2, num_kernel3=args.num_kernel3, num_kernel4=args.num_kernel4, x_dim=5, p_dim=args.D,
                          edge_attr_dim=1, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)
    # # check model size
    print_model_size(model)

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    model = model.to(device)

    # ==========set up optimizer==========
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + \
                str(args.runseed) + '/' + args.filename
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    # ==========training and evaluation==========
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        loss = train(args, model, device, train_loader, optimizer)
        logger.report_scalar(title='training loss', series='Loss', value=loss.item(), iteration=epoch)

        print("====Evaluation")
        if args.eval_train:
            train_enr, train_auc, train_ppv = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_enr = train_auc = train_ppv = 0
        val_enr, val_auc, val_ppv = eval(args, model, device, val_loader)
        test_enr, test_auc, test_ppv = eval(args, model, device, test_loader)

        print("enrichment:train: %f val: %f test: %f" % (train_enr, val_enr, test_enr))
        print("roc auc:train: %f val: %f test: %f" % (train_auc, val_auc, test_auc))
        print("ppv: train: %f val: %f test: %f" % (train_ppv, val_ppv, test_ppv))
        logger.report_scalar(title='enrichment', series='train_acc', value=train_enr, iteration=epoch)
        logger.report_scalar(title='enrichment', series='valid_acc', value=val_enr, iteration=epoch)
        logger.report_scalar(title='enrichment', series='test_acc', value=test_enr, iteration=epoch)

        logger.report_scalar(title='roc_auc', series='train_roc', value=train_auc, iteration=epoch)
        logger.report_scalar(title='roc_auc', series='valid_roc', value=val_auc, iteration=epoch)
        logger.report_scalar(title='roc_auc', series='test_roc', value=test_auc, iteration=epoch)

        logger.report_scalar(title='ppv', series='train_ppv', value=train_ppv, iteration=epoch)
        logger.report_scalar(title='ppv', series='valid_ppv', value=val_ppv, iteration=epoch)
        logger.report_scalar(title='ppv', series='test_ppv', value=test_ppv, iteration=epoch)

        # val_acc_list.append(val_acc)
        # test_acc_list.append(test_acc)
        # train_acc_list.append(train_acc)

        # if not args.filename == "":
        #     writer.add_scalar('data/train auc', train_acc, epoch)
        #     writer.add_scalar('data/val auc', val_acc, epoch)
        #     writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

    model.save_kernellayer('saved_kernellayers')
    torch.save(model.state_dict(), "saved_models/trained_model.pth")


if __name__ == "__main__":
    main()
