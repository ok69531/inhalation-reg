"""
PyTorch-only version of ka_gnn.py.

Main API change:
    old DGL:  out = model(g, features)
    new Torch: out = model(edge_index, features, batch=None, num_nodes=None)

Arguments:
    edge_index: LongTensor with shape [2, E], where edge_index[0] is src and edge_index[1] is dst.
                A tuple/list (src, dst) is also accepted.
    features:   FloatTensor with shape [N, in_feat].
    batch:      Optional LongTensor with shape [N]. batch[i] is the graph id for node i.
                If None, all nodes are pooled as one graph.
    num_nodes:  Optional int. Defaults to features.size(0).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

EdgeIndex = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], list]


def _normalize_edge_index(edge_index: EdgeIndex, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return src, dst LongTensors from edge_index."""
    if isinstance(edge_index, torch.Tensor):
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E].")
        src, dst = edge_index[0], edge_index[1]
    elif isinstance(edge_index, (tuple, list)) and len(edge_index) == 2:
        src, dst = edge_index
    else:
        raise TypeError("edge_index must be a LongTensor [2, E] or a tuple/list (src, dst).")

    src = torch.as_tensor(src, dtype=torch.long, device=device)
    dst = torch.as_tensor(dst, dtype=torch.long, device=device)
    if src.numel() != dst.numel():
        raise ValueError("edge_index src and dst must have the same number of edges.")
    return src, dst


def global_pool(x, batch, pooling):
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if pooling == "sum":
        return global_add_pool(x, batch)
    elif pooling == "avg":
        return global_mean_pool(x, batch)
    elif pooling == "max":
        return global_max_pool(x, batch)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


class KAN_linear(nn.Module):
    def __init__(self, inputdim: int, outdim: int, gridsize: int, addbias: bool = True):
        super().__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) / (math.sqrt(inputdim) * math.sqrt(gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)

        k = torch.arange(1, self.gridsize + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, self.gridsize)
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp).reshape(1, x.shape[0], x.shape[1], self.gridsize)
        s = torch.sin(k * xrshp).reshape(1, x.shape[0], x.shape[1], self.gridsize)

        y = torch.einsum("dbik,djik->bj", torch.cat([c, s], dim=0), self.fouriercoeffs)
        if self.addbias:
            y = y + self.bias
        return y.view(outshape)


class NaiveFourierKANLayer(nn.Module):
    """DGL-free replacement for the original message-passing KAN layer."""

    def __init__(self, in_feats: int, out_feats: int, gridsize: int, addbias: bool = True):
        super().__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, out_feats, in_feats, gridsize) / (math.sqrt(in_feats) * math.sqrt(gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))

    def fourier_transform(self, src_feat: torch.Tensor) -> torch.Tensor:
        """Compute edge messages from source node features. src_feat: [E, in_feats]."""
        k = torch.arange(1, self.gridsize + 1, device=src_feat.device, dtype=src_feat.dtype).view(
            1, 1, 1, self.gridsize
        )
        src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)
        cos_kx = torch.cos(k * src_rshp).reshape(1, src_feat.shape[0], src_feat.shape[1], self.gridsize)
        sin_kx = torch.sin(k * src_rshp).reshape(1, src_feat.shape[0], src_feat.shape[1], self.gridsize)
        return torch.einsum("dbik,djik->bj", torch.cat([cos_kx, sin_kx], dim=0), self.fouriercoeffs)

    def forward(self, edge_index: EdgeIndex, x: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != self.in_feats:
            raise ValueError(f"x must have shape [N, {self.in_feats}].")

        num_nodes = x.size(0) if num_nodes is None else int(num_nodes)
        src, dst = _normalize_edge_index(edge_index, x.device)

        if src.numel() == 0:
            out = x.new_zeros((num_nodes, self.out_feats))
        else:
            if int(src.max()) >= x.size(0) or int(dst.max()) >= num_nodes:
                raise ValueError("edge_index contains node ids outside the valid range.")
            messages = self.fourier_transform(x[src])
            out = x.new_zeros((num_nodes, self.out_feats))
            out.index_add_(0, dst, messages)

        if self.addbias:
            out = out + self.bias
        return out


class KA_GNN_two(nn.Module):
    def __init__(
        self,
        in_feat: int,
        hidden_feat: int,
        out_feat: int,
        out: int,
        grid_feat: int,
        num_layers: int,
        pooling: str,
        use_bias: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
        self.Readout = nn.Sequential(self.linear_1, nn.Sigmoid())

    def forward(
        self,
        edge_index: EdgeIndex,
        h: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        num_nodes = h.size(0) if num_nodes is None else int(num_nodes)
        h = self.kan_line(h)

        for layer in self.layers:
            m = layer(edge_index, h, num_nodes=num_nodes)
            h = torch.nn.functional.leaky_relu(m + h)

        y = global_pool(h, batch=batch, pooling=self.pooling)
        return self.Readout(y)

    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()


class KA_GNN(nn.Module):
    def __init__(
        self,
        in_feat: int,
        hidden_feat: int,
        out_feat: int,
        out: int,
        grid_feat: int,
        num_layers: int,
        pooling: str,
        use_bias: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        self.linear = KAN_linear(hidden_feat, out, grid_feat, addbias=use_bias)

        self.Readout = nn.Sequential(self.linear_1, self.leaky_relu, self.linear_2, nn.Sigmoid())

    def forward(
        self,
        edge_index: EdgeIndex,
        features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        num_nodes = features.size(0) if num_nodes is None else int(num_nodes)
        h = self.kan_line(features)

        for layer in self.layers:
            h = layer(edge_index, h, num_nodes=num_nodes)

        y = global_pool(h, batch=batch, pooling=self.pooling)
        return self.Readout(y)

    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()
