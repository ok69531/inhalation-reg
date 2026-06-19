import sys
sys.path.append('../')

import os
import yaml
import wandb
import logging
import argparse

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from model.ka_gnn_torch import KA_GNN
from modules.utils import set_seed
from utils.common import (
    creat_data,
    CustomDataset,
    collate_fn,
    kagnn_train,
    kagnn_eval
)

logging.basicConfig(format = '', level = logging.INFO)


if torch.cuda.is_available():
    device = torch.device('cuda')
    logging.info('The code uses GPU...')
else:
    device = torch.device('cpu')
    logging.info('The code uses CPU...')


def get_parser():
    parser = argparse.ArgumentParser(description="help")
    parser.add_argument("--tg_num", type=int, default=403)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--encoder_atom', type = str, default = 'cgcnn')
    parser.add_argument('--encoder_bond', type = str, default = 'dim_14')
    parser.add_argument('--grid_feat', type = int, default = 1)
    parser.add_argument('--batch_size', type = int, default = 128)
    return parser
parser = get_parser()
args, _ = parser.parse_known_args()
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])
logging.info(args)


wandb.login(key = open('../wandb_key.txt', 'r').readline())
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val mae'},
    'parameters':{
        'NUM_EPOCHS': {'values': [100, 300, 500]},
        'LR': {'values': [0.001, 0.0005, 0.0001]},
        'pooling': {'values': ['avg', 'max', 'sum']},
        'num_layers': {'values': [2, 4, 6]},
        'hidden_feat': {'values': [32, 64, 128, 256]},
        'out_feat': {'values': [32, 64, 128]},
    }
}
sweep_id = wandb.sweep(sweep_configuration, project = f'TG{args.tg_num}-REG-KAGNN')


def main():
    wandb.init()
    
    args.NUM_EPOCHS = wandb.config.NUM_EPOCHS
    args.LR = wandb.config.LR
    args.pooling = wandb.config.pooling
    args.num_layers = wandb.config.num_layers
    args.hidden_feat = wandb.config.hidden_feat
    args.out_feat = wandb.config.out_feat
    
    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21
    
    inpuit_dim = encode_dim[0] + encode_dim[1]
    target_dim = 1
    
    datafile = f'tg{args.tg_num}'
    creat_data(datafile, encoder_atom, encoder_bond, args.batch_size)
    state = torch.load('dataset/processed/'+datafile+'.pth')

    train_dataset = CustomDataset(state['train_label'], state['train_graph_list'])
    test_dataset = CustomDataset(state['test_label'], state['test_graph_list'])

    val_mae_list, val_mse_list, val_rmse_list, val_r2_list = ([] for _ in range(4))
    for seed in range(3):
        num_train = int(len(train_dataset) * 0.8)
        num_val = len(train_dataset) - num_train
        train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))
        
        set_seed(args.seed)
        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True)
        val_loader = DataLoader(val, batch_size = args.batch_size, shuffle = False)
        logging.info('dataset was loaded!')
        
        LR = args.LR
        NUM_EPOCHS = args.NUM_EPOCHS
        grid_feat = args.grid_feat
        num_layers = args.num_layers
        pooling = args.pooling
        hidden_feat = args.hidden_feat
        out_feat = args.out_feat
        model = KA_GNN(in_feat = inpuit_dim, hidden_feat = hidden_feat, out_feat = out_feat, out=target_dim, 
                       grid_feat = grid_feat, num_layers = num_layers, pooling = pooling, use_bias=True)

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total parameters: {total_params}")

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        
        for epoch in range(1, NUM_EPOCHS+1):
            train_loss = kagnn_train(model, device, train_loader, optimizer, criterion)
            val_loss, val_metrics, _ = kagnn_eval(model, device, val_loader)
            val_mae = val_metrics['log_mae']; val_mse = val_metrics['log_mse']; 
            val_rmse = val_metrics['log_rmse']; val_r2 = val_metrics['log_r2']

            logging.info('=== epoch: {}'.format(epoch))
            logging.info('Train mae: {:.5f} | Validation mae: {:.5f}, mse: {:.5f}, rmse: {:.5f}, r2: {:.5f}'.format(train_loss, val_mae, val_mse, val_rmse, val_r2))
            
            if val_mae < best_val_mae:
                early_stop = 0
                best_val_mae = val_mae
                best_val_mse = val_mse
                best_val_rmse = val_rmse
                best_val_r2 = val_r2
            else:
                early_stop += 1
            
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
    logging.info('KA-GNN')
    logging.info(f'TG {args.tg_num}')

    logging.info('Val MAE: {:.2f} ({:.2f})'.format(np.mean(val_mae_list), np.std(val_mae_list)))
    logging.info('Val MSE: {:.2f} ({:.2f})'.format(np.mean(val_mse_list), np.std(val_mse_list)))
    logging.info('Val RMSE: {:.2f} ({:.2f})'.format(np.mean(val_rmse_list), np.std(val_rmse_list)))
    logging.info('Val R2: {:.2f} ({:.2f})'.format(np.mean(val_r2_list), np.std(val_r2_list)))
    
wandb.agent(sweep_id = sweep_id, function = main, count = 100)
