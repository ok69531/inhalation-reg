import os
import logging
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split

from model.ka_gnn_torch import KA_GNN
from utils.graph_path import path_complex_mol

from rdkit import Chem
from rdkit.Chem import AllChem


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        

        graph = self.graphs[index].to(self.device)
        
        return label, graph
    

def collate_fn(batch):
    labels, graphs = zip(*batch)

    labels = torch.stack(labels)
    batched_graph = Batch.from_data_list(list(graphs))

    return labels, batched_graph


def has_node_with_zero_in_degree(data):
    num_nodes = data.num_nodes

    if num_nodes is None:
        num_nodes = data.x.size(0)

    dst = data.edge_index[1]

    in_degrees = torch.bincount(
        dst,
        minlength=num_nodes,
    )

    return bool((in_degrees == 0).any().item())


def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


def _smiles_screen(raw_data):
    mols = [Chem.MolFromSmiles(x) for x in raw_data.smiles]
    mols_na_idx = [i for i in range(len(mols)) if mols[i] is None]
    data = raw_data.drop(mols_na_idx).reset_index(drop = True)
    if mols_na_idx:
        logging.info(f'Removed SMILES Index: {mols_na_idx}')
    return data


def _preprocess_split_df(split_df, encoder_atom, encoder_bond, split_name):
    """
    split 이후 각 subset에 대해서만 SMILES screening + graph 변환 수행
    """

    label_list = []
    graph_list = []
    smiles_used = []

    for i in range(len(split_df)):
        smiles = split_df.smiles.iloc[i]
        value = split_df.value.iloc[i]

        graph = path_complex_mol(smiles, encoder_atom, encoder_bond)

        if graph is False:
            continue

        if has_node_with_zero_in_degree(graph):
            continue

        label_list.append(torch.tensor(value, dtype=torch.float32))
        graph_list.append(graph)
        smiles_used.append(smiles)

    logging.info(f"{split_name}: {len(graph_list)} graphs were created.")

    return label_list, graph_list, smiles_used


def creat_data(datafile, encoder_atom, encoder_bond, batch_size, test_ratio = 0.2, random_state = 42):
    ''' split 정보 추가 '''

    if datafile == 'tg403':
        datasets = datafile
    elif datafile == 'tg412':
        datasets = 'tg412_413'

    directory_path = 'dataset/processed/'
    os.makedirs(directory_path, exist_ok=True)
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name):
        return True
    
    else:
        df = pd.read_excel('../dataset/raw/' + datasets + '.xlsx')
        df = _smiles_screen(df)
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            shuffle=True,
            random_state=random_state,
            # stratify=stratify_y,
        )
        train_label, train_graph_list, train_smiles = _preprocess_split_df(
            train_df,
            encoder_atom,
            encoder_bond,
            split_name="train",
        )

        test_label, test_graph_list, test_smiles = _preprocess_split_df(
            test_df,
            encoder_atom,
            encoder_bond,
            split_name="test",
        )

        torch.save({
            'train_label': train_label,
            'train_graph_list': train_graph_list,
            'test_label': test_label,
            'test_graph_list': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,  
        }, 'dataset/processed/'+ datafile +'.pth')


def update_node_features(data):
    """
    DGL:
        g.ndata['feat']
        edges.data['feat']
        g.send_and_recv(...)

    PyG:
        data.x
        data.edge_attr
        data.edge_index
    """

    x = data.x                    # node feature: [num_nodes, node_dim]
    edge_index = data.edge_index  # [2, num_edges]
    edge_attr = data.edge_attr    # edge feature: [num_edges, edge_dim]

    num_nodes = data.num_nodes
    edge_dim = edge_attr.size(1)

    # edge_index[0] = source node
    # edge_index[1] = destination node
    dst = edge_index[1]

    # 각 node로 들어오는 edge feature 합산
    agg_feats = torch.zeros(
        num_nodes,
        edge_dim,
        dtype=edge_attr.dtype,
        device=edge_attr.device,
    )

    agg_feats.index_add_(0, dst, edge_attr)

    # 각 node의 incoming edge 개수
    deg = torch.bincount(dst, minlength=num_nodes).to(edge_attr.device)
    deg = deg.clamp(min=1).unsqueeze(1)

    # 평균 aggregation
    agg_feats = agg_feats / deg

    # 기존 node feature와 aggregated edge feature concat
    agg_feats = torch.cat([x, agg_feats], dim=1)

    return agg_feats


def kagnn_train(model, device, train_loader, optimizer, criterion):
    model.train()
    
    loss_list = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        origin_y = data[0].to(device)
        y = torch.log10(origin_y)
        x = update_node_features(data[1]).to(device) 

        out = model(data[1].edge_index, x, data[1].batch, x.size(0)).view(-1)
        # out = model(g, x)

        y = y.to(dtype=out.dtype)
        loss = criterion(out, y)
        loss_list.append(loss.item())
        
        loss.backward()
        optimizer.step()

    return np.average(loss_list)


@torch.no_grad()
def kagnn_eval(model, device, test_loader):
    model.eval()
    
    y, pred = [], []
    origin_y, origin_pred = [], []

    for _, data in enumerate(test_loader):
        origin_batch_y = data[0].to(device)
        batch_y = torch.log10(origin_batch_y)
        
        x = update_node_features(data[1]).to(device)
        output = model(data[1].edge_index, x, data[1].batch, x.size(0)).view(-1)
        
        y.append(batch_y)
        pred.append(output)
        
        origin_y.append(origin_batch_y)
        origin_pred.append(10 ** output)
                
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

