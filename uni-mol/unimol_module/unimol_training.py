import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def unimol_train(model, optimizer, scheduler, device, loader, criterion):
    model.train()
    
    loss_list = []
    
    for batch in loader:
        tokens = batch["tokens"].to(device)
        distances = batch["distances"].to(device)
        edge_types = batch["edge_types"].to(device)
        labels = torch.log10(batch["labels"]).to(device)
        
        scores, _ = model(tokens, distances, edge_types)
        
        loss = criterion(scores.view(-1), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # record
        loss_list.append(loss.item())
    
    return np.average(loss_list)        


@torch.no_grad()
def unimol_evaluation(model, device, loader):
    model.eval()
    
    y, pred = [], []
    origin_y, origin_pred = [], []
    
    for _, batch in enumerate(loader):
        tokens = batch["tokens"].to(device)
        distances = batch["distances"].to(device)
        edge_types = batch["edge_types"].to(device)
        labels = torch.log10(batch["labels"]).to(device)
        
        scores, _ = model(tokens, distances, edge_types)
        
        # record
        y.append(labels)
        pred.append(scores.view(-1))
        
        origin_y.append(batch['labels'])
        origin_pred.append(10 ** scores.view(-1))
        
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
