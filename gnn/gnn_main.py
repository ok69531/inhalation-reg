import sys
sys.path.append('../')

import os
import yaml
import logging
import argparse
import warnings

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


parser = argparse.ArgumentParser()
parser.add_argument('--gnn_type', type = str, default = None, help = 'gcn, gin')
parser.add_argument('--tg_num', type = int, default = 403, help = '403, 412')
parser.add_argument('--train_frac', type = float, default = 0.8)
parser.add_argument('--val_frac', type = float, default = 0.1)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--readout', type = str, default = None)
parser.add_argument('--hidden_dim', type = int, default = None)
parser.add_argument('--num_layers', type = int, default = None)
parser.add_argument('--skip_con', type = str, default = None, help = 'None, all, last')
parser.add_argument('--lr', type = float, default = None)
parser.add_argument('--epochs', type = int, default = None)
parser.add_argument('--optimizer', type = str, default = None)
parser.add_argument('--weight_decay', type = float, default = None)
parser.add_argument('--seed', type = int, default = 42)

args, unknown = parser.parse_known_args()

with open('best_hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)
hparams = hparams[args.tg_num][args.gnn_type]
parser.set_defaults(**hparams)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


def main():
    logging.info('')
    logging.info(args)

    train_dataset = InhaleRegDataset(root = '../dataset', tg_num = args.tg_num, split = 'train')
    test_dataset = InhaleRegDataset(root = '../dataset', tg_num = args.tg_num, split = 'test')

    set_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(train_dataset)):
        avg_nodes += train_dataset[i].x.shape[0]
        avg_edge_index += train_dataset[i].edge_index.shape[1]

    avg_nodes /= len(train_dataset)
    avg_edge_index /= len(train_dataset)
    logging.info('graphs {}, avg_nodes {:.4f}, avg_edge_index {:.4f}'.format(len(train_dataset), avg_nodes, avg_edge_index/2))
    
    criterion = nn.L1Loss(reduction = 'mean')
    output_dim = train_dataset.num_classes
    model = GraphNeuralNetwork(output_dim, args).to(device)
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
    final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10

    for epoch in range(1, args.epochs+1):
        train_loss = gnn_train(model, optimizer, device, train_loader, criterion)
        val_loss, val_metric, _ = gnn_evaluation(model, device, train_loader, criterion)

        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train MAE: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, R2: {:.5f}'.format(
                            val_loss, val_metric['log_mse'], val_metric['log_rmse'], val_metric['log_r2']))

        if val_loss < best_val_mae:
            _, test_metric, test_pred = gnn_evaluation(model, device, test_loader, criterion)
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
    save_path = os.path.join(save_path, f'{args.gnn_type}.pt')
    torch.save(checkpoints, save_path)
    
    logging.info('')
    logging.info('Model: {}'.format(args.gnn_type))
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
