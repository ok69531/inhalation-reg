import sys
sys.path.append('../')

import os
import yaml
import wandb
import logging
import argparse
import warnings

import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import Adadelta
from torch.utils.data import DataLoader, random_split
# from torch_geometric.loader import DataLoader

from modules.utils import set_seed
from modules.ml_dataset import load_dataset
from modules.dl_dataset import InhaleRegDataset

from ifm_mlp import (
    MyDataset,
    collate_fn,
    IFMMLP,
    ifm_train,
    ifm_eval
)

warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


parser = argparse.ArgumentParser()
parser.add_argument('--fp_type', type = str, default = 'maccs', help = 'maccs, morgan, rdkit, pattern, layered')
parser.add_argument('--tg_num', type = int, default = 403, help = '403, 412')
parser.add_argument('--train_frac', type = float, default = 0.8)
parser.add_argument('--val_frac', type = float, default = 0.1)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--k', type = int, default = None)
parser.add_argument('--sigma', type = float, default = None)
parser.add_argument('--hidden_units0', type = int, default = None)
parser.add_argument('--hidden_units1', type = int, default = None)
parser.add_argument('--hidden_units2', type = int, default = None)
parser.add_argument('--dropout', type = float, default = None)
# parser.add_argument('--lr', type = float, default = None)
parser.add_argument('--epochs', type = int, default = None)
parser.add_argument('--weight_decay', type = float, default = None)
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
        'fp_type': {'values': [args.fp_type]},
        'k': {'values': [4, 8, 16, 32]}, 
        'sigma': {'values': [1, 3, 6]}, 
        
        'epochs': {'values': [100, 300]},
        # 'lr': {'values': [0.005, 0.001]},
        'weight_decay': {'values': [0, 1e-5, 1e-3, 0.01]},
        'dropout': {'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        
        'hidden_units0': {'values': [32, 64, 128, 256]},
        'hidden_units1': {'values': [32, 64, 128, 256]},
        'hidden_units2': {'values': [32, 64, 128, 256]},
    }
}
# gnn type, skip_con, tg num은 sh 파일에서 튜닝

sweep_id = wandb.sweep(sweep_configuration, project = f'TG{args.tg_num}-REG-IFM')


def main():
    wandb.init()
    
    args.fp_type = wandb.config.fp_type
    args.k = wandb.config.k
    args.sigma = wandb.config.sigma
    args.epochs = wandb.config.epochs
    args.weight_decay = wandb.config.weight_decay
    args.dropout = wandb.config.dropout
    args.hidden_units0 = wandb.config.hidden_units0
    args.hidden_units1 = wandb.config.hidden_units1
    args.hidden_units2 = wandb.config.hidden_units2

    logging.info('')
    logging.info(args)
    
    x, y, smiles = load_dataset(
        root = '../dataset',
        tg_num = args.tg_num,
        fp_type = args.fp_type,
        log_transform = True
    )
    train_idx, test_idx = train_test_split(range(len(y)), test_size = 0.2, random_state = args.seed)
    x_tr = x[train_idx]; x_te = x[test_idx]
    y_tr = y[train_idx]; y_te = y[test_idx]
    
    train_dataset = MyDataset(x_tr, y_tr)
    # test_dataset = MyDataset(x_te, y_te)
    
    val_mae_list, val_mse_list, val_rmse_list, val_r2_list = ([] for _ in range(4))
    for seed in range(3):
        num_train = int(len(train_dataset) * 0.8)
        num_val = len(train_dataset) - num_train
        train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))
        
        set_seed(args.seed)
        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn, generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(val, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

        # elif args.model == 'gnn':
        #     train_dataset = InhaleRegDataset(root = '../dataset', tg_num = args.tg_num, split = 'train')
        #     test_dataset = InhaleRegDataset(root = '../dataset', tg_num = args.tg_num, split = 'test')

        #     set_seed(args.seed)
        #     train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True)
        #     test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
        
        # avg_nodes = 0.0
        # avg_edge_index = 0.0
        # for i in range(len(train_dataset)):
        #     avg_nodes += train_dataset[i].x.shape[0]
        #     avg_edge_index += train_dataset[i].edge_index.shape[1]

        # avg_nodes /= len(train_dataset)
        # avg_edge_index /= len(train_dataset)
        # logging.info('graphs {}, avg_nodes {:.4f}, avg_edge_index {:.4f}'.format(len(train_dataset), avg_nodes, avg_edge_index/2))
        
        criterion = nn.L1Loss(reduction = 'none')
        input_dim = len(train_dataset[0][0])
        output_dim = 1
        model = IFMMLP(input_dim, output_dim, args).to(device)
        optimizer = Adadelta(model.parameters(), weight_decay = args.weight_decay)

        best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10

        for epoch in range(1, args.epochs+1):
            train_loss = ifm_train(model, optimizer, device, train_loader, criterion)
            val_loss, val_metric, _ = ifm_eval(model, device, val_loader)
            val_mae = val_metric['log_mae']; val_mse = val_metric['log_mse']; val_rmse = val_metric['log_rmse']; val_r2 = val_metric['log_r2']

            logging.info('=== epoch: {}'.format(epoch))
            logging.info('Train MAE: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, R2: {:.5f}'.format(
                                val_loss, val_metric['log_mse'], val_metric['log_rmse'], val_metric['log_r2']))

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
    logging.info('Model: IFM-MLP')
    logging.info('TG: {}'.format(args.tg_num))
    logging.info('Fingerprints: {}'.format(args.fp_type))

    logging.info('Val MAE: {:.2f} ({:.2f})'.format(np.mean(val_mae_list), np.std(val_mae_list)))
    logging.info('Val MSE: {:.2f} ({:.2f})'.format(np.mean(val_mse_list), np.std(val_mse_list)))
    logging.info('Val RMSE: {:.2f} ({:.2f})'.format(np.mean(val_rmse_list), np.std(val_rmse_list)))
    logging.info('Val R2: {:.2f} ({:.2f})'.format(np.mean(val_r2_list), np.std(val_r2_list)))


wandb.agent(sweep_id = sweep_id, function = main, count = 200)
