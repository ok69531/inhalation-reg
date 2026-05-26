import math
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class IFMEncoding(nn.Module):
    def __init__(self, n_features: int, k: int = 8, sigma: float = 6.0):
        super().__init__()
        self.n_features = n_features
        self.k = k
        self.c = nn.Parameter(torch.empty(k))
        nn.init.normal_(self.c, mean=0.0, std=sigma)

    def forward(self, x):
        # x: (B, d)
        # v: (B, k, d)
        v = 2 * math.pi * self.c.view(1, self.k, 1) * x.unsqueeze(1)

        # z: (B, 2k, d)
        z = torch.cat([torch.sin(v), torch.cos(v)], dim=1)

        # MLP에 넣기 위해 flatten: (B, 2*k*d)
        return z.flatten(start_dim=1)


class IFMMLP(nn.Module):
    def __init__(self, input_dim, output_dim, args):
    # def __init__(self, inputs, hidden_units, outputs, k=8, sigma=6.0, dropout=0.1):
        super().__init__()
        self.embedding = IFMEncoding(input_dim, k=args.k, sigma=args.sigma)

        in_dim = 2 * args.k * input_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, args.hidden_units0),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_units0, args.hidden_units1),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_units1, args.hidden_units2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_units2, output_dim),
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.net(x)


class MyDataset(object):
    def __init__(self, Xs, Ys):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.masks = torch.tensor(~np.isnan(Ys) * 1.0, dtype=torch.float32)
        # convert np.nan to 0
        self.Ys = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32)


    def __len__(self):
        return len(self.Ys)

    def __getitem__(self, idx):

        X = self.Xs[idx]
        Y = self.Ys[idx]
        mask = self.masks[idx]

        return X, Y, mask


def ifm_train(model, optimizer, device, loader, criterion):
    model.train()

    loss_list = []
    
    for i, batch in enumerate(loader):
        xs, ys, masks = batch
        xs = xs.to(device)
        ys = ys.to(device)
        masks = masks.to(device)
        
        scores = model(xs)
        loss = (criterion(scores.view(-1), ys) * (masks != 0).float()).mean()
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record
        loss_list.append(loss.item())
    
    return np.average(loss_list)


@torch.no_grad()
def ifm_eval(model, device, loader):
    model.eval()
    
    y, pred = [], []
    origin_y, origin_pred = [], []
    
    for _, batch in enumerate(loader):
        xs, ys, masks = batch
        xs = xs.to(device)
        ys = ys.to(device)
        masks = masks.to(device)
        
        scores = model(xs)
        
        # record
        y.append(ys)
        pred.append(scores.view(-1))
        
        origin_y.append(10 ** ys)
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


def collate_fn(data_batch):
    Xs, Ys, masks = map(list, zip(*data_batch))

    Xs = torch.stack(Xs, dim=0)
    Ys = torch.stack(Ys, dim=0)
    masks = torch.stack(masks, dim=0)

    return Xs, Ys, masks
