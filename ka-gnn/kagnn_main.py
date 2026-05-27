"""
@author: longlee
"""

import sys
sys.path.append('../')

import os
import yaml
import logging
import argparse

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
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


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    with open('best_hparams.yaml', 'r') as f:
        hparams = yaml.safe_load(f)
    hparams = hparams[args.tg_num]
    parser.set_defaults(**hparams)
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    logging.info(args)
    
    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21
    
    target_dim = 1
    inpuit_dim = encode_dim[0] + encode_dim[1]
    
    datafile = f'tg{args.tg_num}'
    creat_data(datafile, encoder_atom, encoder_bond, args.batch_size)
    state = torch.load('dataset/processed/'+datafile+'.pth')

    train_dataset = CustomDataset(state['train_label'], state['train_graph_list'])
    test_dataset = CustomDataset(state['test_label'], state['test_graph_list'])

    set_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, collate_fn=collate_fn)
    logging.info('dataset was loaded!')
    
    LR = args.LR
    NUM_EPOCHS = args.NUM_EPOCHS
    grid_feat = args.grid_feat
    num_layers = args.num_layers
    pooling = args.pooling
    hidden_feat = args.hidden_feat
    out_feat = args.out_feat
    model = KA_GNN(in_feat = inpuit_dim, hidden_feat = hidden_feat, out_feat = out_feat, out=target_dim, 
                    grid_feat = grid_feat, num_layers = num_layers, pooling = pooling, use_bias=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
    final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10
    
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = kagnn_train(model, device, train_loader, optimizer, criterion)
        val_loss, val_metric, _ = kagnn_eval(model, device, train_loader)

        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train MAE: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, R2: {:.5f}'.format(
                    val_loss, val_metric['log_mse'], val_metric['log_rmse'], val_metric['log_r2']))
        
        if val_loss < best_val_mae:
            _, test_metric, test_pred = kagnn_eval(model, device, test_loader)
            
            best_val_mae = val_loss
            best_val_mse = val_metric['log_mse']
            best_val_rmse = val_metric['log_rmse']
            best_val_r2 = val_metric['log_r2']
            final_test_mae = test_metric['log_mae']
            final_test_mse = test_metric['log_mse']
            final_test_rmse = test_metric['log_rmse']
            final_test_r2 = test_metric['log_r2']
            
            params = deepcopy(model.state_dict())
            optim_params = deepcopy(optimizer.state_dict())

    checkpoints = {
        'params_dict': params,
        'optim_dict': optim_params,
        'metric': test_metric,
        'pred_result': test_pred
    }
    
    save_path = f'saved_result/tg{args.tg_num}'
    if not os.path.isdir(save_path): os.makedirs(save_path)
    save_path = os.path.join(save_path, f'kagnn.pt')
    torch.save(checkpoints, save_path)
    
    logging.info('')
    logging.info('Model: {}'.format('KA-GNN'))
    logging.info('TG: {}'.format(args.tg_num))

    logging.info('')
    logging.info(f"Log-scaled Test MAE: {final_test_mae:.5f}")
    logging.info(f"Log-scaled Test MSE: {final_test_mse:.5f}")
    logging.info(f"Log-scaled Test RMSE: {final_test_rmse:.5f}")
    logging.info(f"Log-scaled Test R2: {final_test_r2:.5f}")
    
    logging.info(f"Original Scale Test MAE: {test_metric['origin_mae']:.5f}")
    logging.info(f"Original Scale Test MSE: {test_metric['origin_mse']:.5f}")
    logging.info(f"Original Scale Test RMSE: {test_metric['origin_rmse']:.5f}")
    logging.info(f"Original Scale Test R2: {test_metric['origin_r2']:.5f}")    


if __name__ == '__main__':
    main()