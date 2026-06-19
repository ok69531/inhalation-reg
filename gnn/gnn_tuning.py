import sys
sys.path.append('../')

import os
import logging
import argparse
import warnings

import wandb

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from modules.utils import set_seed
from modules.dl_dataset import InhaleRegDataset

from gnn_model.gnn import (
    GraphNeuralNetwork,
    gnn_train,
    gnn_evaluation
)

warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)

device = torch.device('cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


parser = argparse.ArgumentParser()
parser.add_argument('--gnn_type', type = str, default = 'gcn', help = 'gcn, gin')
parser.add_argument('--skip_con', type = str, default = None, help = 'None, all, last')
parser.add_argument('--tg_num', type = int, default = 403, help = '403, 412')
parser.add_argument('--train_frac', type = float, default = 0.8)
parser.add_argument('--val_frac', type = float, default = 0.1)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--hidden_dim', type = int, default = 128)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--readout', type = str, default = 'mean')
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--optimizer', type = str, default = 'adam')
parser.add_argument('--weight_decay', type = float, default = 0)
parser.add_argument('--seed', type = int, default = 42)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])



wandb.login(key = open('../wandb_key.txt', 'r').readline())
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val mae'},
    'parameters':{
        'gnn_type': {'values': [args.gnn_type]}, 
        'skip_con': {'values': [args.skip_con]}, 
        
        'epochs': {'values': [100, 300]},
        'lr': {'values': [0.005, 0.001]},
        'weight_decay': {'values': [0, 1e-5]},
        
        'hidden_dim': {'values': [32, 64, 128, 256]},
        'num_layers': {'values': [2, 3, 5]},
        'readout': {'values': ['sum', 'mean', 'max']},
        'optimizer': {'values': ['adam', 'sgd']}
    }
}
# gnn type, skip_con, tg num은 sh 파일에서 튜닝

sweep_id = wandb.sweep(sweep_configuration, project = f'TG{args.tg_num}-REG-GNN')

def main():
    wandb.init()
    
    args.epochs = wandb.config.epochs
    args.lr = wandb.config.lr
    args.weight_decay = wandb.config.weight_decay
    args.hidden_dim = wandb.config.hidden_dim
    args.num_layers = wandb.config.num_layers
    args.readout = wandb.config.readout
    args.optimizer = wandb.config.optimizer
    
    logging.info('')
    logging.info(args)

    train_dataset = InhaleRegDataset(root = '../dataset', tg_num = args.tg_num, split = 'train')
    test_dataset = InhaleRegDataset(root = '../dataset', tg_num = args.tg_num, split = 'test')
    
    # avg_nodes = 0.0
    # avg_edge_index = 0.0
    # for i in range(len(train_dataset)):
    #     avg_nodes += train_dataset[i].x.shape[0]
    #     avg_edge_index += train_dataset[i].edge_index.shape[1]

    # avg_nodes /= len(train_dataset)
    # avg_edge_index /= len(train_dataset)
    # logging.info('graphs {}, avg_nodes {:.4f}, avg_edge_index {:.4f}'.format(len(train_dataset), avg_nodes, avg_edge_index/2))
    
    val_mae_list, val_mse_list, val_rmse_list, val_r2_list = ([] for _ in range(4))
    for seed in range(3):
        num_train = int(len(train_dataset) * 0.8)
        num_val = len(train_dataset) - num_train
        train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))
        
        set_seed(args.seed)
        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True)
        val_loader = DataLoader(val, batch_size = args.batch_size, shuffle = False)
        
        criterion = nn.L1Loss(reduction = 'mean')
        output_dim = train_dataset.num_classes
        model = GraphNeuralNetwork(output_dim, args)
        if args.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10

        for epoch in range(1, args.epochs+1):
            train_loss = gnn_train(model, optimizer, device, train_loader, criterion)
            val_loss, val_metric, _ = gnn_evaluation(model, device, val_loader, criterion)
            val_mae = val_metric['log_mae']; val_mse = val_metric['log_mse']; val_rmse = val_metric['log_rmse']; val_r2 = val_metric['log_r2']
            
            if val_loss < best_val_mae:
                early_stop = 0
                best_val_mae = val_mae
                best_val_mse = val_mse
                best_val_rmse = val_rmse
                best_val_r2 = val_r2
            else:
                early_stop += 1
            
            logging.info('=== epoch: {}'.format(epoch))
            logging.info('Train mae: {:.5f} | Validation mae: {:.5f}, mse: {:.5f}, rmse: {:.5f}, r2: {:.5f}'.format(train_loss, val_mae, val_mse, val_rmse, val_r2))
            
            if early_stop > 50: break
        
        val_mae_list.append(best_val_mae)
        val_mse_list.append(best_val_mse)
        val_rmse_list.append(best_val_rmse)
        val_r2_list.append(best_val_r2)
        
    wandb.log({
        'val mae': np.mean(val_mae_list),
        'val mse': np.mean(val_mse_list),
        'val rmse': np.mean(val_rmse_list),
        'val r2': np.mean(val_r2_list),
    })
    
    logging.info('')
    logging.info('Model: {}'.format(args.gnn_type))
    logging.info('TG: {}'.format(args.tg_num))

    logging.info('Val MAE: {:.2f} ({:.2f})'.format(np.mean(val_mae_list), np.std(val_mae_list)))
    logging.info('Val MSE: {:.2f} ({:.2f})'.format(np.mean(val_mse_list), np.std(val_mse_list)))
    logging.info('Val RMSE: {:.2f} ({:.2f})'.format(np.mean(val_rmse_list), np.std(val_rmse_list)))
    logging.info('Val R2: {:.2f} ({:.2f})'.format(np.mean(val_r2_list), np.std(val_r2_list)))


wandb.agent(sweep_id = sweep_id, function = main, count = 300)
