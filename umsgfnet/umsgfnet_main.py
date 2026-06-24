import sys
sys.path.append('../')

import os
import yaml
import warnings
import logging
import argparse

import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.optim as optim
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
parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
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

args, unknown = parser.parse_known_args()

with open('best_hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)
hparams = hparams[args.tg_num]
parser.set_defaults(**hparams)

try: args = parser.parse_args()
except: args = parser.parse_args([])


def main():
    logging.info('')
    logging.info(args)
    
    train_dataset = MoleculeDataset(root = '../dataset', tg_num = args.tg_num, split = 'train')
    test_dataset = MoleculeDataset(root = '../dataset', tg_num = args.tg_num, split = 'test')

    set_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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
        val_loss, val_metric, _ = eval_reg(model, device, train_loader, 'train')
        
        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train MAE: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, R2: {:.5f}'.format(
                            val_loss, val_metric['log_mse'], val_metric['log_rmse'], val_metric['log_r2']))

        if val_loss < best_val_mae:
            _, test_metric, test_pred = eval_reg(model, device, test_loader, 'test')
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
    save_path = os.path.join(save_path, f'umsgfnet.pt')
    torch.save(checkpoints, save_path)
    
    logging.info('')
    logging.info('Model: UMSGFNet')
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


if __name__ == "__main__":
    main()

