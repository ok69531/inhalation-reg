import sys
sys.path.append('../')

import os
import yaml
import logging
import argparse
import warnings

from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
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
args, unknown = parser.parse_known_args()

with open('best_hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)
hparams = hparams[args.tg_num]
parser.set_defaults(**hparams)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


def main():
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
    test_dataset = UniMolPropertyDataset(data["test"])

    set_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True, collate_fn=unimol_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn=unimol_collate_fn)

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
        val_loss, val_metric, _ = unimol_evaluation(model, device, train_loader)
        
        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train MAE: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, R2: {:.5f}'.format(
                            val_loss, val_metric['log_mse'], val_metric['log_rmse'], val_metric['log_r2']))
        if val_loss < best_val_mae:
            _, test_metric, test_pred = unimol_evaluation(model, device, test_loader)
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
    save_path = os.path.join(save_path, f'unimol.pt')
    torch.save(checkpoints, save_path)
    
    logging.info('')
    logging.info('Model: Uni-Mol')
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
