import sys
sys.path.append('../')

import os
import wandb
import logging
import argparse
import warnings

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from transformers import get_polynomial_decay_schedule_with_warmup

from modules.utils import set_seed

from unimol_module.load_dataset import (
    DEFAULT_UNIMOL_MOL_DICT,
    load_or_preprocess_unimol_excel,
    UniMolPropertyDataset,
    unimol_collate_fn
)
from unimol_module.unimol_model import (
    UniMolForMolecularPropertyPrediction, 
    UniMolConfig
)
from unimol_module.unimol_training import (
    unimol_train,
    unimol_evaluation
)

warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


parser = argparse.ArgumentParser()
parser.add_argument('--tg_num', type = int, default = 403, help = '403, and 412')
parser.add_argument('--train_frac', type = float, default = 0.8)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--epochs', type = int, default = 40)
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
        'epochs': {'values': [40, 60]},
        'lr': {'values': [5e-5, 8e-5, 1e-4, 4e-4, 5e-4]},
        'batch_size': {'values': [32, 128]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.5]},
        'warmup_ratio': {'values': [0.0, 0.06, 0.1] },
        'weight_decay': {'values': [0, 1e-4]}
    }
}
sweep_id = wandb.sweep(sweep_configuration, project = f'TG{args.tg_num}-REG-UniMol')


def main():
    wandb.init()
    
    args.epochs = wandb.config.epochs
    args.lr = wandb.config.lr
    args.batch_size = wandb.config.batch_size
    args.dropout = wandb.config.dropout
    args.warmup_ratio = wandb.config.warmup_ratio
    args.weight_decay = wandb.config.weight_decay
    
    logging.info('')
    logging.info(args)

    EXCEL_PATH = "../dataset/raw"
    CACHE_PATH = "dataset/processed"

    SMILES_COL = "smiles"
    LABEL_COLS = "value"

    dictionary = DEFAULT_UNIMOL_MOL_DICT

    data = load_or_preprocess_unimol_excel(
        tg_num=args.tg_num,
        excel_path=EXCEL_PATH,
        cache_path=CACHE_PATH,
        smiles_col=SMILES_COL,
        label_cols=LABEL_COLS,
        dictionary=dictionary,
    )

    train_dataset = UniMolPropertyDataset(data["train"])
    # test_dataset = UniMolPropertyDataset(data["test"])

    val_mae_list, val_mse_list, val_rmse_list, val_r2_list = ([] for _ in range(4))
    for seed in range(3):
        num_train = int(len(train_dataset) * 0.8)
        num_val = len(train_dataset) - num_train
        train, val = random_split(train_dataset, lengths = [num_train, num_val], generator=torch.Generator().manual_seed(seed))

        set_seed(args.seed)
        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True, collate_fn=unimol_collate_fn)
        val_loader = DataLoader(val, batch_size = args.batch_size, shuffle = False, collate_fn=unimol_collate_fn)

        criterion = nn.L1Loss(reduction = 'mean')
        config = UniMolConfig()
        model = UniMolForMolecularPropertyPrediction(config, dropout=args.dropout, include_pretraining_heads=False).to(device)
        checkpoints = torch.load('pretrained_ckp.pt', map_location=device)
        model.load_state_dict(checkpoints['model'], strict=False)
        optimizer = Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        num_training_steps = len(train_loader) * args.epochs
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=int(num_training_steps * args.warmup_ratio)
        )

        best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10
        
        for epoch in range(1, args.epochs + 1):
            train_loss = unimol_train(model, optimizer, scheduler, device, train_loader, criterion)
            val_loss, val_metric, _ = unimol_evaluation(model, device, val_loader)
            
            logging.info('=== epoch: {}'.format(epoch))
            logging.info('Train MAE: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, R2: {:.5f}'.format(
                                val_loss, val_metric['log_mse'], val_metric['log_rmse'], val_metric['log_r2']))
            if val_loss < best_val_mae:
                early_stop = 0
                best_val_mae = val_loss
                best_val_mse = val_metric['log_mse']
                best_val_rmse = val_metric['log_rmse']
                best_val_r2 = val_metric['log_r2']
                # params = deepcopy(model.state_dict())
                # optim_params = deepcopy(optimizer.state_dict())
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
    logging.info('UniMol')
    logging.info(f'TG {args.tg_num}')

    logging.info('Val MAE: {:.2f} ({:.2f})'.format(np.mean(val_mae_list), np.std(val_mae_list)))
    logging.info('Val MSE: {:.2f} ({:.2f})'.format(np.mean(val_mse_list), np.std(val_mse_list)))
    logging.info('Val RMSE: {:.2f} ({:.2f})'.format(np.mean(val_rmse_list), np.std(val_rmse_list)))
    logging.info('Val R2: {:.2f} ({:.2f})'.format(np.mean(val_r2_list), np.std(val_r2_list)))


wandb.agent(sweep_id = sweep_id, function = main, count = 100)
