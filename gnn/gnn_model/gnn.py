import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

from .conv_layer import GINConv, GCNConv
from .encoder import AtomEncoder

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import warnings
warnings.filterwarnings('ignore')


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


class GraphNeuralNetwork(nn.Module):
    def __init__(self, output_dim, args):
        super(GraphNeuralNetwork, self).__init__()
        
        self.gnn_type = args.gnn_type
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.skip_con = args.skip_con
        self.readout_layer = get_readout_layers(args.readout)
        
        # graph embedding layer (GNN)
        self.atom_encoder = AtomEncoder(emb_dim = self.hidden_dim)
        
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if args.gnn_type == 'gcn':
                self.convs.append(GCNConv(self.hidden_dim))
            elif args.gnn_type == 'gin':
                self.convs.append(GINConv(self.hidden_dim))
            else:
                raise ValueError('Invalid GNN Layer.')
        
        # classifier
        self.lin = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        initial_emb = self.atom_encoder(x)
        node_features = initial_emb
        if self.skip_con == 'all':
            for conv in self.convs:
                node_features = conv(node_features, edge_index, edge_attr)
                node_features = node_features + initial_emb
        elif self.skip_con == 'last':
            for conv in self.convs:
                node_features = conv(node_features, edge_index, edge_attr)
            node_features = node_features + initial_emb
        else:
            for conv in self.convs:
                node_features = conv(node_features, edge_index, edge_attr)

        # graph embedding
        for readout in self.readout_layer:
            graph_feature = readout(node_features, batch)

        scores = self.lin(graph_feature)
        # probs = self.softmax(logits)

        return scores, graph_feature


def gnn_train(model, optimizer, device, loader, criterion):
    model.train()
    
    loss_list = []
    
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        
        scores, graph_feature = model(batch)
        loss = criterion(scores.view(-1), batch.y)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record
        loss_list.append(loss.item())
    
    return np.average(loss_list)


@torch.no_grad()
def gnn_evaluation(model, device, loader, criterion):
    model.eval()
    
    y, pred = [], []
    origin_y, origin_pred = [], []
    
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        scores, graph_feature = model(batch)
        
        # record
        y.append(batch.y)
        pred.append(scores.view(-1))
        
        origin_y.append(batch.origin_y)
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
