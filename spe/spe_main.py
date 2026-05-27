import sys
sys.path.append('../')

import os
import yaml
import logging
import warnings
from functools import partial
from copy import deepcopy

import numpy as np

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from modules.utils import set_seed
from modules.dl_dataset import InhaleRegDataset

from spe_argument import load_spe_args
from spe_trainer import (
    create_mlp,
    create_mlp_ln,
    # get_snorm,
    calc_eigh,
    get_param_groups,
    lr_lambda,
    training,
    evaluation
)
from spe_model.spe_model import construct_model


warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')


args = load_spe_args()
logging.info(args)
with open('best_hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)
hparams = hparams[args.tg_num]
vars(args).update(hparams)


def main():
    args.base_hidden_dims = args.node_emb_dims
    args.phi_hidden_dims = args.node_emb_dims
    args.mlp_hidden_dims = args.node_emb_dims
    
    train_dataset = InhaleRegDataset(root = args.data_path, tg_num = args.tg_num, split = 'train')
    test_dataset = InhaleRegDataset(root = args.data_path, tg_num = args.tg_num, split = 'test')
    train_dataset = calc_eigh(train_dataset, args)
    test_dataset = calc_eigh(test_dataset, args)
    # test_dataset = log_transform_target(test_dataset)

    set_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, generator=torch.Generator().manual_seed(args.seed), drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    kwargs = {}
    kwargs['residual'] = args.residual
    kwargs['feature_type'] = 'discrete'

    criterion = nn.L1Loss(reduction = 'mean')
    
    model = construct_model(args, create_mlp, **kwargs)
    model.to(device)

    param_groups = get_param_groups(model, args)
    n_total_steps = len(train_loader) * args.n_epochs
    optimizer = optim.Adam(param_groups, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=partial(lr_lambda, args=args, n_total_steps=n_total_steps)
    )

    best_val_mae, best_val_mse, best_val_rmse, best_val_r2 = 1e+10, 1e+10, 1e+10, -1e+10
    final_test_mae, final_test_mse, final_test_rmse, final_test_r2 = 1e+10, 1e+10, 1e+10, -1e+10

    for epoch in range(1, args.n_epochs + 1):
        train_loss = training(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_metrics, _ = evaluation(model, train_loader, criterion, device, args)
        val_mae = val_metrics['log_mae']; val_mse = val_metrics['log_mse']
        val_rmse = val_metrics['log_rmse']; val_r2 = val_metrics['log_r2']
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            
            _, test_metric, test_pred = evaluation(model, device, test_loader, criterion)
            final_test_mae = test_metric['log_mae']
            final_test_mse = test_metric['log_mse']
            final_test_rmse = test_metric['log_rmse']
            final_test_r2 = test_metric['log_r2']
            
            model_params = deepcopy(model.state_dict())
            optim_params = deepcopy(optimizer.state_dict())
        
        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train mae: {:.5f} | Validation mae: {:.5f}, mse: {:.5f}, rmse: {:.5f}, r2: {:.5f}'.format(train_loss, val_mae, val_mse, val_rmse, val_r2))
        
    checkpoints = {
        'params_dict': model_params,
        'optim_dict': optim_params,
        'metric': test_metric,
        'pred_result': test_pred
    }
    
    save_path = f'saved_result/tg{args.tg_num}'
    if not os.path.isdir(save_path): os.makedirs(save_path)
    save_path = os.path.join(save_path, f'spe.pt')
    torch.save(checkpoints, save_path)
    
    logging.info('')
    logging.info('SPE')
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
