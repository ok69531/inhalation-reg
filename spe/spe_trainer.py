import numpy as np

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import get_laplacian, to_dense_adj

from spe_model.spe_mlp import MLP

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def create_mlp(in_dims: int, out_dims: int, args) -> MLP:
    return MLP(in_dims, out_dims, args, args.mlp_use_bn)


def create_mlp_ln(in_dims: int, out_dims: int, args) -> MLP:
    return MLP(in_dims, out_dims, args, args.mlp_use_ln)


# def get_snorm(instance: Data) -> Data:
#     # get the graph normalization for nodes on the fly
#     size = instance.num_nodes
#     snorm = torch.FloatTensor(size, 1).fill_(1./float(size)).sqrt()
#     instance.update({"snorm": snorm})
#     return instance


# def calc_eigh(instance: Data, args) -> Data:
#     # get spectrum
#     n = instance.num_nodes
#     L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
#     L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)              # [N, N]

#     Lambda = torch.zeros(1, args.pe_dims)   # [1, D_pe]
#     V = torch.zeros(n, args.pe_dims)        # [N, D_pe]

#     d = min(n, args.pe_dims)   # number of eigen-pairs to use (then we zero-pad up to D_pe)
#     eigenvalues, eigenvectors = torch.linalg.eigh(L)   # [N], [N, N]
#     Lambda[0, :d] = eigenvalues[0:d]
#     V[:, :d] = eigenvectors[:, 0:d]

#     instance.update({"Lambda": Lambda, "V": V})
    
#     snorm = torch.FloatTensor(n, 1).fill_(1./float(n)).sqrt()
#     instance.update({"snorm": snorm})

#     return instance

def calc_eigh_per_instance(instance, args):
    try:
        n = instance.num_nodes
    except:
        n = instance.x.size(0)
    
    L_edge_index, L_values = get_laplacian(instance.edge_index, normalization = 'sym', num_nodes = n)
    L = to_dense_adj(L_edge_index, edge_attr = L_values, max_num_nodes = n).squeeze(dim = 0)
    
    Lambda = torch.zeros(1, args.pe_dims)
    V = torch.zeros(n, args.pe_dims)
    
    d = min(n, args.pe_dims)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    Lambda[0, :d] = eigenvalues[0:d]
    V[:, :d] = eigenvectors[:, 0:d]
    
    instance.update({'Lambda': Lambda, 'V': V})
    
    snorm = torch.FloatTensor(n, 1).fill_(1./float(n)).sqrt()
    instance.update({'snorm': snorm})
    
    return instance


def calc_eigh(obj, args):
    # leaf: PyG Data면 계산 적용
    if isinstance(obj, Data):
        return calc_eigh_per_instance(obj, args)
    
    if isinstance(obj, InMemoryDataset):
        return [calc_eigh(v, args) for v in obj]

    # dict면 value들을 재귀 처리
    if isinstance(obj, dict):
        return {k: calc_eigh(v, args) for k, v in obj.items()}

    # list면 원소들을 재귀 처리
    if isinstance(obj, list):
        return [calc_eigh(v, args) for v in obj]

    # tuple도 들어올 수 있으면 (선택)
    if isinstance(obj, tuple):
        return tuple(calc_eigh(v, args) for v in obj)

    # 그 외 타입은 그대로
    return obj


def get_param_groups(model, args):
    return [{
        "name": name,
        "params": [param],
        "weight_decay": 0.0 if "bias" in name else args.weight_decay
    } for name, param in model.named_parameters()]


def lr_lambda(curr_step: int, *, args, n_total_steps: int) -> float:
    if curr_step < args.n_warmup_steps:
        return float(curr_step) / float(max(1, args.n_warmup_steps))
    return max(
        0.0,
        float(n_total_steps - curr_step) / float(max(1, n_total_steps - args.n_warmup_steps))
    )


def training(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += (loss.item() * batch.y.size(0))
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluation(model, loader, criterion, device, args, original_scale = False):
    model.eval()
    
    total_loss = 0
    y, pred = [], []
    origin_y, origin_pred = [], []
    
    for batch in loader:
        batch = batch.to(device)
        
        score = model(batch)
        
        loss = criterion(score, batch.y)
        total_loss += (loss * batch.y.size(0))
        
        y.append(batch.y)
        pred.append(score.view(-1))
        
        origin_y.append(batch.origin_y)
        origin_pred.append(10 ** score.view(-1))
    
    total_loss /= len(loader.dataset)
    y = torch.cat(y).cpu().numpy()
    pred = torch.cat(pred, dim = 0).cpu().detach().numpy()
    origin_y = torch.cat(origin_y).cpu().numpy()
    origin_pred = torch.cat(origin_pred, dim = 0).cpu().detach().numpy()
    
    subgraph_metric = {
        'log_mae': mean_absolute_error(y, pred), 
        'log_mse': mean_squared_error(y, pred), 
        'log_rmse': np.sqrt(mean_squared_error(y, pred)), 
        'log_r2': r2_score(y, pred), 
        'origin_mae': mean_absolute_error(origin_y, origin_pred), 
        'origin_mse': mean_squared_error(origin_y, origin_pred), 
        'origin_rmse': np.sqrt(mean_squared_error(origin_y, origin_pred)), 
        'origin_r2': r2_score(origin_y, origin_pred)
    }
    pred_result = {
        'log_y_test': y,
        'log_pred': pred,
        'origin_y_test': origin_y,
        'origin_pred': origin_pred
    }
    
    return subgraph_metric['log_mae'], subgraph_metric, pred_result
    