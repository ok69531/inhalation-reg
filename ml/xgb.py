import sys
sys.path.append('../')

import logging
import warnings
import argparse

import numpy as np
from tqdm import tqdm
from rdkit import RDLogger

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import mutual_info_regression

from common import (
    ParameterGrid,
    find_best_model,
    save_results
)
from modules.utils import set_seed
from modules.ml_dataset import load_dataset


warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
logging.basicConfig(format='', level=logging.INFO)


def load_args():
    parser = argparse.ArgumentParser()
    
    # data arguments
    parser.add_argument('--data_path', default = '../dataset', type = str)
    parser.add_argument('--assay_name', default = None, type = str)
    parser.add_argument('--tg_num', default = None, type = int, help = '403, 412')
    parser.add_argument('--test_size', default = 0.2, type = float)
    parser.add_argument('--random_state', default = 42, type = int)
    parser.add_argument('--fp_type', default = 'maccs', type = str, help = 'maccs, morgan, rdkit, layered, pattern')
    parser.add_argument('--log_transform', default = True, type = bool)
    
    # learning arguments
    parser.add_argument('--use_smogn', default = False, type = bool)
    parser.add_argument('--use_feat_sel', default = False, type = bool)
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    return args
    

def main():
    args = load_args()
    logging.info(args)

    if args.tg_num is None:
        save_path = f'saved_result/{args.assay_name}'
    else: 
        save_path = f'saved_result/tg{args.tg_num}'
    
    x, y, smiles = load_dataset(
        root = args.data_path,
        tg_num = args.tg_num,
        fp_type = args.fp_type,
        log_transform = args.log_transform
    )
    
    train_idx, test_idx = train_test_split(range(len(y)), test_size = args.test_size, random_state = args.random_state)
    x_tr = x[train_idx]; x_te = x[test_idx]
    y_tr = y[train_idx]; y_te = y[test_idx]
    
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)
    
    if args.use_feat_sel:
        mi = mutual_info_regression(x_tr, y_tr, random_state = args.random_state)
        n_keep = max(1, int(0.30 * x_tr.shape[1]))
        top_idx = np.argsort(mi)[-n_keep:]

        x_tr = x_tr[:, top_idx]
        x_te = x_te[:, top_idx]
    
    params_dict = {
        "n_estimators": [5, 10, 50, 100, 150, 300],
        "learning_rate": [0.01, 0.03, 0.1],
        "max_depth": [3, 4, 6],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
        'objective': ['reg:absoluteerror']
    }
    params = ParameterGrid(params_dict)
    logging.info(f'The number of hyperparameter combinations:{len(params)}')
    
    result = {'model': {}, 'mae': {}, 'mse': {}, 'rmse': {}, 'r2': {}}
    kf = KFold(n_splits = 5, shuffle = True, random_state = args.random_state)
    
    for p in tqdm(range(len(params))):
        model_key = f'model{p}'
        result['model'][model_key] = params[p]
        result['mae'][model_key] = []
        result['mse'][model_key] = []
        result['rmse'][model_key] = []
        result['r2'][model_key] = []
        
        for train_idx, val_idx in kf.split(x_tr, y_tr):
            fold_tr_x, fold_val_x = x_tr[train_idx], x_tr[val_idx]
            fold_tr_y, fold_val_y = y_tr[train_idx], y_tr[val_idx]
            
            model = XGBRegressor(random_state = args.random_state, **params[p])
            model.fit(fold_tr_x, fold_tr_y)
            pred = model.predict(fold_val_x)
            
            result['mae'][model_key].append(mean_absolute_error(fold_val_y, pred))
            result['mse'][model_key].append(mean_squared_error(fold_val_y, pred))
            result['rmse'][model_key].append(np.sqrt(mean_squared_error(fold_val_y, pred)))
            result['r2'][model_key].append(r2_score(fold_val_y, pred))
        
        save_results(result, path = save_path, file_name = f'gbt_{args.fp_type}_feat_sel_{args.use_feat_sel}.pkl')
            
    best_model_key, best_params, best_r2 = find_best_model(result, metric = 'mae')
    
    best_mae = np.mean(result['mae'][best_model_key])
    best_mse = np.mean(result['mse'][best_model_key])
    best_rmse = np.mean(result['rmse'][best_model_key])
    
    logging.info(f'Best Model Parameters: {best_params}')    
    logging.info(f'Log-scaled Validation MAE: {best_mae:.5f}')    
    logging.info(f'Log-scaled Validation MSE: {best_mse:.5f}')    
    logging.info(f'Log-scaled Validation RMSE: {best_rmse:.5f}')    
    
    final_model = XGBRegressor(random_state = args.random_state, **best_params)
    final_model.fit(x_tr, y_tr)
    pred = final_model.predict(x_te)
    
    log_test_mae = mean_absolute_error(y_te, pred)
    log_test_mse = mean_squared_error(y_te, pred)
    log_test_rmse = np.sqrt(mean_squared_error(y_te, pred))
    log_test_r2 = r2_score(y_te, pred)
    
    pred_origin = 10 ** pred
    y_te_origin = 10 ** y_te
    origin_test_mae = mean_absolute_error(y_te_origin, pred_origin)
    origin_test_mse = mean_squared_error(y_te_origin, pred_origin)
    origin_test_rmse = np.sqrt(mean_squared_error(y_te_origin, pred_origin))
    origin_test_r2 = r2_score(y_te_origin, pred_origin)
    
    test_metric = {
        'log_mae': log_test_mae,
        'log_mse': log_test_mse,
        'log_rmse': log_test_rmse,
        'log_r2': log_test_r2,
        'origin_mae': origin_test_mae,
        'origin_mse': origin_test_mse,
        'origin_rmse': origin_test_rmse,
        'origin_r2': origin_test_r2,
    }
    
    pred_result = {
        'log_y_test': y_te,
        'log_pred': pred,
        'origin_y_test': y_te_origin,
        'origin_pred': pred_origin
    }
    
    logging.info(f"Log-scaled Test MAE: {log_test_mae:.5f}")
    logging.info(f"Log-scaled Test MSE: {log_test_mse:.5f}")
    logging.info(f"Log-scaled Test RMSE: {log_test_rmse:.5f}")
    logging.info(f"Log-scaled Test R2: {log_test_r2:.5f}")
    
    logging.info(f"Original Scale Test MAE: {origin_test_mae:.5f}")
    logging.info(f"Original Scale Test MSE: {origin_test_mse:.5f}")
    logging.info(f"Original Scale Test RMSE: {origin_test_rmse:.5f}")
    logging.info(f"Original Scale Test R2: {origin_test_r2:.5f}")
    
    checkpoints = {
        'params': final_model.get_params(),
        'metric': test_metric,
        'pred_result': pred_result
    }
    
    file_name = f'best_gbt_{args.fp_type}_feat_sel_{args.use_feat_sel}.json'
    save_results(checkpoints, path = save_path, file_name = file_name)
    
    logging.info(f"Best model saved with MAE: {origin_test_mae:.5f}")


if __name__ == '__main__':
    main()
