import os
import pickle
import numpy as np
from itertools import product
from collections.abc import Iterable


def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v)))

    return params_grid


def find_best_model(results, metric='mae', metric_agg='mean'):
    best_model = None
    best_score = -np.inf
    best_model_key = None
    
    for model_key in results['model'].keys():
        scores = results[metric][model_key]
        if metric_agg == 'mean':
            agg_score = np.mean(scores)
        elif metric_agg == 'median':
            agg_score = np.median(scores)
        else:
            raise ValueError("metric_agg must be either 'mean' or 'median'")
        
        if agg_score > best_score:
            best_score = agg_score
            best_model = results['model'][model_key]
            best_model_key = model_key
    
    return best_model_key, best_model, best_score


def save_results(result, path, file_name):
    if not os.path.exists(path): os.makedirs(path)
    with open(os.path.join(path, file_name), 'wb') as f:
        pickle.dump(result, f)
