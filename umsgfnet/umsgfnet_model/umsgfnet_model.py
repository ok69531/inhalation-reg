import sys
sys.path.append('../')

import re
import ast
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from .fast_kan import FastKAN as KAN
from umsgfnet_module.featurization import BatchMolGraph


# ---------------------------------------------------------------------------------
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, True)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


# --------------------------------------------------------------------


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim=0, index=index.view(-1))
    target = target.view(final_size)
    return target


class SimpleDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleDecoder, self).__init__()
        self.decoder_fc = nn.Linear(hidden_size, hidden_size)
    def forward(self, latent):
        return self.decoder_fc(latent)


class GraphAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(GraphAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, mol_graph):
        latent = self.encoder(mol_graph)
        reconstructed_graph = self.decoder(latent)
        return reconstructed_graph
    def loss(self, original_graph, reconstructed_graph):
        f_atoms, _, _, _, _, _, _ = original_graph.get_components()
        f_atoms = f_atoms.to(self.device)
        return F.mse_loss(reconstructed_graph, f_atoms)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    def forward(self, z1, z2):
        sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = F.cross_entropy(sim / self.temperature, labels)
        return loss


class MultiModalPool(nn.Module):
    def __init__(self, hidden_size, pool_type='mean'):
        super(MultiModalPool, self).__init__()
        self.hidden_size = hidden_size
        self.pool_type = pool_type
    def forward(self, inputs):
        if self.pool_type == 'mean':
            return torch.mean(inputs, dim=0)
        elif self.pool_type == 'sum':
            return torch.sum(inputs, dim=0)
        elif self.pool_type == 'max':
            return torch.max(inputs, dim=0)[0]
        else:
            raise ValueError("Invalid pool_type. Choose from 'mean', 'sum', or 'max'.")


# --------------------------------------------------------------------


class UMSGFNetEncoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size, depth, device):
        super(UMSGFNetEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.device = device
        self.bias = False

        self.dropout_layer = nn.Dropout(p=0.2)
        self.act_func = nn.LeakyReLU(negative_slope=0.01)

        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        self.W_i = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias)
        self.W_h_local = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.global_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=False)
        self.W_o = KAN([self.atom_fdim + self.hidden_size, self.hidden_size])
        self.W_fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, mol_graph, return_atom_hiddens: bool = False):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device), f_bonds.to(self.device),
            a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)
        )

        inputs = self.W_i(f_bonds)
        message = self.act_func(inputs)

        for _ in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)   # [n_atoms, max_deg, hidden]
            a_message = nei_a_message.sum(dim=1)            # [n_atoms, hidden]
            rev_message = message[b2revb]
            message = a_message[b2a] - rev_message
            message = self.W_h_local(message)
            message = self.act_func(inputs + message)
            message = self.dropout_layer(message)

        nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)

        atom_hiddens_local = self.act_func(self.W_o(a_input))
        atom_hiddens_local = self.dropout_layer(atom_hiddens_local)

        atom_hiddens_global, _ = self.global_attention(
            atom_hiddens_local.unsqueeze(0),
            atom_hiddens_local.unsqueeze(0),
            atom_hiddens_local.unsqueeze(0)
        )
        atom_hiddens_global = atom_hiddens_global.squeeze(0)

        atom_hiddens = torch.cat([atom_hiddens_local, atom_hiddens_global], dim=1)
        atom_hiddens = self.act_func(self.W_fusion(atom_hiddens))

        mol_vecs = []
        for (a_start, a_size) in a_scope:
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens.mean(dim=0)
                mol_vecs.append(mol_vec)
        mol_vecs = torch.stack(mol_vecs, dim=0)

        if return_atom_hiddens:
            return mol_vecs, atom_hiddens, a_scope
        return mol_vecs


class MemoryModule(nn.Module):
    def __init__(self, memory_size, hidden_size):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.memory = Parameter(torch.randn(memory_size, hidden_size))
        self.attention = nn.Linear(hidden_size, memory_size)
    def forward(self, query):
        attention_weights = F.softmax(self.attention(query), dim=-1)
        memory_output = torch.matmul(attention_weights, self.memory)
        return memory_output


class UMSGFNetEncoderWithMemory(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size, depth, device, memory_size=128):
        super(UMSGFNetEncoderWithMemory, self).__init__()
        self.memory_module = MemoryModule(memory_size, hidden_size)
        self.encoder = UMSGFNetEncoder(atom_fdim, bond_fdim, hidden_size, depth, device)
    def forward(self, mol_graph, return_atom_hiddens: bool = False):
        if return_atom_hiddens:
            ligand_x, atom_hiddens, a_scope = self.encoder(mol_graph, return_atom_hiddens=True)
            memory_output = self.memory_module(ligand_x)
            combined_output = ligand_x + memory_output
            return combined_output, atom_hiddens, a_scope
        else:
            ligand_x = self.encoder(mol_graph)
            memory_output = self.memory_module(ligand_x)
            combined_output = ligand_x + memory_output
            return combined_output


class WeightFusion(nn.Module):
    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    def forward(self, inputs: Tensor) -> Tensor:
        return sum([inputs[i] * weight for i, weight in enumerate(self.weight[0][0])]) + self.bias


# --------------------------------------

def _maybe_strip_bytes_literal(s: str) -> str:
    if isinstance(s, (bytes, bytearray)):
        try:
            return s.decode("utf-8")
        except Exception:
            return str(s)
    if isinstance(s, str):
        m = re.fullmatch(r"""b['"](.+?)['"]""", s.strip())
        if m:
            return m.group(1)
    return s


def _maybe_parse_list_string(s: str):
    if not isinstance(s, str):
        return s
    st = s.strip()
    if st.startswith('[') and st.endswith(']'):
        try:
            v = ast.literal_eval(st)
            return v
        except Exception:
            return s
    return s


def _normalize_smis(smis):
    if smis is None:
        return []
    if hasattr(smis, "tolist"):
        try:
            smis = smis.tolist()
        except Exception:
            smis = str(smis)
    if isinstance(smis, (list, tuple)):
        cleaned = []
        for x in smis:
            x = _maybe_strip_bytes_literal(x)
            x = _maybe_parse_list_string(x)
            if isinstance(x, (list, tuple)):
                if len(x) > 0 and all(isinstance(t, str) and len(t) == 1 for t in x):
                    cleaned.append(''.join(x))
                elif len(x) == 1 and isinstance(x[0], str):
                    cleaned.append(x[0])
                else:
                    cleaned.append(''.join(str(t) for t in x))
            else:
                if isinstance(x, (bytes, bytearray)):
                    try:
                        cleaned.append(x.decode("utf-8"))
                    except Exception:
                        cleaned.append(str(x))
                else:
                    cleaned.append(str(x))
        if len(cleaned) > 0 and all(len(s) == 1 for s in cleaned):
            return [''.join(cleaned)]
        return cleaned
    if isinstance(smis, str):
        s = _maybe_strip_bytes_literal(smis)
        parsed = _maybe_parse_list_string(s)
        if isinstance(parsed, (list, tuple)):
            return _normalize_smis(parsed)
        else:
            return [str(parsed)]
    return [str(smis)]

# -------------------------------------------------------------------

class UMSGFNet(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim, fp_fdim, device='cpu', out_dim=2, memory_size=128):
        super(UMSGFNet, self).__init__()
        
        self.tg_num = args.tg_num
        self.device = device
        
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim
        
        hidden_size = args.hidden_dim
        depth = args.depth

        self.encoder = UMSGFNetEncoderWithMemory(self.atom_fdim, self.bond_fdim, hidden_size, depth, device, memory_size)

        self.mlp_fp = nn.Sequential(
            KAN([fp_fdim, 2048]),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, hidden_size)
        )
        self.feature_fusion = WeightFusion(2, hidden_size, device=device)
        self.mlp = nn.Linear(hidden_size, out_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=False)

        self.decoder = SimpleDecoder(hidden_size)
        self.graph_autoencoder = GraphAutoencoder(self.encoder, self.decoder, device=self.device)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.multimodal_pool = MultiModalPool(hidden_size)

    # @property
    # def split_tag(self):
    #     """
    #     model.train()이면 'train',
    #     model.eval()이면 'test'
    #     """
    #     return "train" if self.training else "test"
    
    def forward(self, batch, split_tag, self_supervised=False):
        smis = _normalize_smis(getattr(batch, "smi", None))
        mol_batch = BatchMolGraph(smis, atom_fdim=self.atom_fdim, bond_fdim=self.bond_fdim,
                                  fp_fdim=self.fp_fdim, tg_num = self.tg_num, split_tag = split_tag)

        ligand_x = self.encoder.forward(mol_batch)                         # [B, H]
        fp_x = self.mlp_fp(mol_batch.fp_x.to(self.device).to(torch.float32))  # [B, H]

        ligand_x_ = ligand_x.unsqueeze(0)  # [1,B,H]
        fp_x_ = fp_x.unsqueeze(0)          # [1,B,H]
        attn_output, _ = self.attention(ligand_x_, fp_x_, fp_x_)           # [1,B,H]
        ligand_x = attn_output.squeeze(0)                                  # [B,H]

        ligand_x = self.feature_fusion(torch.stack([ligand_x, fp_x], dim=0))  # [B,H]
        x = self.mlp(ligand_x)  # [B, out_dim]

        if self_supervised:
            reconstructed_graph = self.graph_autoencoder(mol_batch)
            reconstruction_loss = self.graph_autoencoder.loss(mol_batch, reconstructed_graph)
            contrastive_loss = self.contrastive_loss(ligand_x, fp_x)
            total_loss = reconstruction_loss + contrastive_loss
            return x, total_loss

        return x, ligand_x

    def forward_with_explanations(self, batch, split_tag):
        smis = _normalize_smis(getattr(batch, "smi", None))
        mol_batch = BatchMolGraph(smis, atom_fdim=self.atom_fdim, bond_fdim=self.bond_fdim,
                                  fp_fdim=self.fp_fdim, tg_num = self.tg_num, split_tag = split_tag)

        ligand_x, atom_hiddens, a_scope = self.encoder.forward(
            mol_batch, return_atom_hiddens=True
        )  # ligand_x: [B,H]; atom_hiddens: [N_atoms,H]

        fp_raw = mol_batch.fp_x.to(self.device).to(torch.float32)  # [B, 6338]（原始拼接）
        fp_x = self.mlp_fp(fp_raw)  # [B, H]    （供模型内部使用）

        attn_output, _ = self.attention(ligand_x.unsqueeze(0), fp_x.unsqueeze(0), fp_x.unsqueeze(0))
        ligand_x = attn_output.squeeze(0)
        ligand_x = self.feature_fusion(torch.stack([ligand_x, fp_x], dim=0))
        logits = self.mlp(ligand_x)

        return logits, atom_hiddens, a_scope, fp_raw

    @torch.no_grad()
    def logits_from_atom_hiddens(self, atom_hiddens: torch.Tensor, a_scope, fp_raw: torch.Tensor):
        """
        - atom_hiddens: [N_atoms_total, H]
        - a_scope:      list[(start, size)]
        - fp_raw:       [B, 6338]
        """
        device = atom_hiddens.device
        H = atom_hiddens.size(-1)

        # 聚合原子到分子级表示  [B,H]
        mol_vecs = []
        for (a_start, a_size) in a_scope:
            if a_size > 0:
                mol_vecs.append(atom_hiddens.narrow(0, a_start, a_size).mean(dim=0))
            else:
                mol_vecs.append(torch.zeros(H, device=device))
        ligand_x = torch.stack(mol_vecs, dim=0)  # [B,H]

        # 原始指纹 → 过同一个 MLP，得到与训练一致的 fp_x
        fp_x = self.mlp_fp(fp_raw.to(device).to(torch.float32))  # [B,H]

        # 注意力 + 融合 + 预测头（与 forward 保持一致）
        attn_output, _ = self.attention(ligand_x.unsqueeze(0), fp_x.unsqueeze(0), fp_x.unsqueeze(0))
        ligand_x = attn_output.squeeze(0)
        ligand_x = self.feature_fusion(torch.stack([ligand_x, fp_x], dim=0))
        logits = self.mlp(ligand_x)
        return logits
