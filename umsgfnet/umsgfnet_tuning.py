import sys
sys.path.append('../')

import os
import yaml
import warnings
import logging
import argparse

import wandb

import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from umsgfnet_model.umsgfnet_model import UMSGFNet
from umsgfnet_model.training import train_reg, eval_reg
from umsgfnet_module.load_dataset import MoleculeDataset
from umsgfnet_module.featurization import get_atom_fdim, get_bond_fdim

from modules.utils import set_seed


warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


parser = argparse.ArgumentParser(description='PyTorch implementation of UMSGFNet with per-epoch case analysis')
parser.add_argument('--tg_num', type=int, default=403, help='403 or 412')

parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001 , help='learning rate for the prediction layer')
parser.add_argument('--weight_decay', type=float, default=1e-3 , help='learning rate for the prediction layer')
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--depth', type=int, default=7, help="the depth of molecule encoder")

parser.add_argument('--seed', type=int, default=42, help="seed for splitting the dataset")
parser.add_argument('--huber_beta', type=float, default=7.0, help='Beta parameter for Huber loss')
parser.add_argument('--weight_power', type=float, default=1.0, help='Power for error-based weighting in Weighted Huber Loss')
parser.add_argument('--weight_epsilon', type=float, default=1e-6, help='Epsilon for error-based weighting in Weighted Huber Loss')
try: args = parser.parse_args()
except: args = parser.parse_args([])


wandb.login(key = open('../wandb_key.txt', 'r').readline())
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val mae'},
    'parameters':{
        'batch_size': {'values': [64, 128]}, 
        
        'epochs': {'values': [100, 300]},
        'lr': {'values': [0.001, 0.0005, 0.0001]},
        'weight_decay': {'values': [0, 1e-3, 1e-5]},
        
        'hidden_dim': {'values': [512, 256, 128]},
        'depth': {'values': [2, 3, 5, 7]},
        
        'huber_beta': {'values': [0.5, 1., 3.]},
        'weight_power': {'values': [0., 1.]}
    }
}
# gnn type, skip_con, tg num은 sh 파일에서 튜닝

sweep_id = wandb.sweep(sweep_configuration, project = f'TG{args.tg_num}-REG-UMSGFNet')


def main():
    wandb.init()
    
    args.batch_size = wandb.config.batch_size
    args.epochs = wandb.config.epochs
    args.lr = wandb.config.lr
    args.weight_decay = wandb.config.weight_decay
    args.hidden_dim = wandb.config.hidden_dim
    args.depth = wandb.config.depth
    args.huber_beta = wandb.config.huber_beta
    args.weight_power = wandb.config.weight_power
    
    logging.info('')
    logging.info(args)
    
    train_dataset = MoleculeDataset(root = '../dataset', tg_num = args.tg_num, split = 'train')
    # test_dataset = MoleculeDataset(root = '../dataset', tg_num = args.tg_num, split = 'test')

    val_mae_list, val_mse_list, val_rmse_list, val_r2_list = ([] for _ in range(4))
    for seed in range(3):
        num_train = int(len(train_dataset) * 0.8)
        num_val = len(train_dataset) - num_train
        train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))
        
        set_seed(args.seed)
        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True)
        val_loader = DataLoader(val, batch_size = args.batch_size, shuffle = False)

        atom_dim = get_atom_fdim()
        bond_dim = get_bond_fdim()
        out_dim = 1
        model = UMSGFNet(args, atom_fdim=atom_dim, bond_fdim=bond_dim, fp_fdim=6338, device=device, out_dim=out_dim)
        model.to(device)

        model_param_group = []
        model_param_group.append({"params": model.parameters(), "lr": args.lr})
        optimizer = optim.Adam(model_param_group, weight_decay=args.weight_decay)

        # fixed_example_batch = None
        # for _b in train_loader:
        #     fixed_example_batch = _b
        #     break
        # if fixed_example_batch is None:
        #     raise RuntimeError("训练集为空，无法进行案例分析")

        # example_index = max(0, min(args.example_index, len(test_dataset) - 1))
        # example_data = test_dataset[example_index]

        best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_reg(args, model, device, train_loader, optimizer)
            val_loss, val_metric, _ = eval_reg(model, device, val_loader, 'train')
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
    logging.info('Model: UMSGFNet')
    logging.info('TG: {}'.format(args.tg_num))

    logging.info('Val MAE: {:.2f} ({:.2f})'.format(np.mean(val_mae_list), np.std(val_mae_list)))
    logging.info('Val MSE: {:.2f} ({:.2f})'.format(np.mean(val_mse_list), np.std(val_mse_list)))
    logging.info('Val RMSE: {:.2f} ({:.2f})'.format(np.mean(val_rmse_list), np.std(val_rmse_list)))
    logging.info('Val R2: {:.2f} ({:.2f})'.format(np.mean(val_r2_list), np.std(val_r2_list)))


wandb.agent(sweep_id = sweep_id, function = main, count = 300)
