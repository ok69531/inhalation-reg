# https://github.com/nleroy917/drugclip/tree/master/unimol

from typing import Optional, Tuple, Literal, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Config
# -------------------------

class UniMolConfig:
    def __init__(
        self,
        vocab_size: int = 31,
        hidden_size: int = 512,
        num_hidden_layers: int = 15,
        num_attention_heads: int = 64,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        activation_dropout: float = 0.0,
        gbf_kernels: int = 128,
        padding_idx: int = 0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.activation_dropout = activation_dropout
        self.gbf_kernels = gbf_kernels
        self.padding_idx = padding_idx


# -------------------------
# Utils
# -------------------------

def gaussian_rbf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    pi = 3.14159265359
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


# -------------------------
# Checkpoint-compatible modules
# -------------------------

class GaussianLayer(nn.Module):
    """
    Checkpoint keys:
        gbf.means.weight: [1, 128]
        gbf.stds.weight: [1, 128]
        gbf.mul.weight: [961, 1]
        gbf.bias.weight: [961, 1]
    """

    def __init__(self, num_kernels: int = 128, num_edge_types: int = 961):
        super().__init__()
        self.K = num_kernels
        self.means = nn.Embedding(1, num_kernels)
        self.stds = nn.Embedding(1, num_kernels)
        self.mul = nn.Embedding(num_edge_types, 1)
        self.bias = nn.Embedding(num_edge_types, 1)

        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, distances: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        mul = self.mul(edge_types).type_as(distances)
        bias = self.bias(edge_types).type_as(distances)

        x = mul * distances.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5

        return gaussian_rbf(x.float(), mean, std).type_as(self.means.weight)


class NonLinearHead(nn.Module):
    """
    Used for:
        gbf_proj.linear1.weight: [128, 128]
        gbf_proj.linear2.weight: [64, 128]

        pair2coord_proj.linear1.weight: [64, 64]
        pair2coord_proj.linear2.weight: [1, 64]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        hidden: Optional[int] = None,
    ):
        super().__init__()
        hidden = hidden or in_features
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)

        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Checkpoint keys:
        encoder.layers.N.self_attn.in_proj.weight: [1536, 512]
        encoder.layers.N.self_attn.out_proj.weight: [512, 512]
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 64, dropout: float = 0.1):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q * self.scaling

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        if attn_bias is not None:
            # attn_bias: [B * H, N, N] or [B, H, N, N]
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.view(bsz, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights + attn_bias

        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask[:, None, None, :],
                torch.finfo(attn_weights.dtype).min,
            )

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(out)

        # 다음 layer에 pair/bias처럼 넘길 수 있도록 [B*H, N, N]
        pair_rep = attn_weights.contiguous().view(bsz * self.num_heads, seq_len, seq_len)

        return out, pair_rep


class TransformerEncoderLayer(nn.Module):
    """
    Checkpoint keys:
        encoder.layers.N.self_attn.*
        encoder.layers.N.self_attn_layer_norm.*
        encoder.layers.N.fc1.*
        encoder.layers.N.fc2.*
        encoder.layers.N.final_layer_norm.*
    """

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.self_attn_layer_norm(x)
        x, pair_rep = self.self_attn(
            x,
            attn_bias=attn_bias,
            padding_mask=padding_mask,
        )
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        return x, pair_rep


class TransformerEncoder(nn.Module):
    """
    Checkpoint-compatible name:
        encoder.*

    Checkpoint keys:
        encoder.emb_layer_norm.weight: [512]
        encoder.final_layer_norm.weight: [512]
        encoder.final_head_layer_norm.weight: [64]
        encoder.layers.0~14.*
    """

    def __init__(self, config: UniMolConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads

        self.emb_layer_norm = nn.LayerNorm(config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        # checkpoint에 존재하므로 로딩 호환용으로 둔다.
        # property prediction forward에서는 직접 쓰지 않는다.
        self.final_head_layer_norm = nn.LayerNorm(config.num_attention_heads)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=config.hidden_size,
                    ffn_dim=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    attention_dropout=config.attention_probs_dropout_prob,
                    activation_dropout=config.activation_dropout,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        x = self.emb_layer_norm(x)
        x = self.dropout(x)

        if padding_mask is not None:
            x = x * (~padding_mask).unsqueeze(-1).type_as(x)

            attn_bias = attn_bias.view(bsz, self.num_heads, seq_len, seq_len)
            attn_bias = attn_bias.masked_fill(
                padding_mask[:, None, None, :],
                torch.finfo(attn_bias.dtype).min,
            )
            attn_bias = attn_bias.view(bsz * self.num_heads, seq_len, seq_len)

        for layer in self.layers:
            x, attn_bias = layer(
                x,
                attn_bias=attn_bias,
                padding_mask=None,
            )

        x = self.final_layer_norm(x)
        return x


# -------------------------
# Optional pretraining heads
# -------------------------

class MaskLMHead(nn.Module):
    """
    로딩 호환용.
    Checkpoint keys:
        lm_head.weight: [31, 512]
        lm_head.bias: [31]
        lm_head.dense.weight: [512, 512]
        lm_head.layer_norm.weight: [512]
    """

    def __init__(self, hidden_size: int = 512, vocab_size: int = 31):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return F.linear(x, self.weight, self.bias)


class DistanceHead(nn.Module):
    """
    로딩 호환용.
    Checkpoint keys:
        dist_head.dense.weight: [64, 64]
        dist_head.layer_norm.weight: [64]
        dist_head.out_proj.weight: [1, 64]
    """

    def __init__(self, num_heads: int = 64):
        super().__init__()
        self.dense = nn.Linear(num_heads, num_heads)
        self.layer_norm = nn.LayerNorm(num_heads)
        self.out_proj = nn.Linear(num_heads, 1)

    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        x = self.dense(pair_rep)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x


# -------------------------
# Property prediction head
# -------------------------

class UniMolPredictionHead(nn.Module):
    """
    공식 Uni-Mol property prediction 방식:
        features[:, 0, :] -> Dropout -> Linear -> activation -> Dropout -> Linear
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_outputs: int = 1,
        inner_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "tanh",
        pooling: Literal["cls", "mean"] = "cls",
    ):
        super().__init__()

        inner_dim = inner_dim or hidden_size

        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, inner_dim)
        self.out_proj = nn.Linear(inner_dim, num_outputs)
        self.activation_fn = get_activation(activation)

    def pool(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return features[:, 0, :]

        if self.pooling == "mean":
            if padding_mask is None:
                return features.mean(dim=1)

            valid = (~padding_mask).float().unsqueeze(-1)
            return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pool(features, padding_mask=padding_mask)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# -------------------------
# Final model
# -------------------------

class UniMolForMolecularPropertyPrediction(nn.Module):
    """
    이 클래스의 state_dict key는 checkpoint와 맞게 설계됨.

    checkpoint에서 바로 맞는 key:
        embed_tokens.*
        encoder.*
        gbf.*
        gbf_proj.*
        lm_head.*
        pair2coord_proj.*
        dist_head.*

    새로 학습되는 key:
        prediction_head.*
    """

    def __init__(
        self,
        config: Optional[UniMolConfig] = None,
        num_outputs: int = 1,
        problem_type: Literal[
            "regression",
            "single_label_classification",
            "binary_classification",
            "multi_label_classification",
        ] = "regression",
        inner_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "tanh",
        pooling: Literal["cls", "mean"] = "cls",
        freeze_backbone: bool = False,
        include_pretraining_heads: bool = False,
    ):
        super().__init__()

        self.config = config or UniMolConfig()
        self.padding_idx = self.config.padding_idx
        self.problem_type = problem_type
        self.num_outputs = num_outputs

        # checkpoint top-level key와 맞춤
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.encoder = TransformerEncoder(self.config)

        num_edge_types = self.config.vocab_size * self.config.vocab_size
        self.gbf = GaussianLayer(
            num_kernels=self.config.gbf_kernels,
            num_edge_types=num_edge_types,
        )

        self.gbf_proj = NonLinearHead(
            in_features=self.config.gbf_kernels,
            out_features=self.config.num_attention_heads,
            activation="gelu",
            hidden=self.config.gbf_kernels,
        )

        # checkpoint 로딩 호환용. property forward에서는 사용하지 않음.
        if include_pretraining_heads:
            self.lm_head = MaskLMHead(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
            )
            self.pair2coord_proj = NonLinearHead(
                in_features=self.config.num_attention_heads,
                out_features=1,
                activation="relu",
                hidden=self.config.num_attention_heads,
            )
            self.dist_head = DistanceHead(
                num_heads=self.config.num_attention_heads,
            )

        self.prediction_head = UniMolPredictionHead(
            hidden_size=self.config.hidden_size,
            num_outputs=num_outputs,
            inner_dim=inner_dim,
            dropout=dropout,
            activation=activation,
            pooling=pooling,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        backbone_names = [
            "embed_tokens",
            "encoder",
            "gbf",
            "gbf_proj",
        ]

        for name in backbone_names:
            module = getattr(self, name)
            for p in module.parameters():
                p.requires_grad = False

    def get_dist_features(
        self,
        distances: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        distances:
            [B, N, N]

        edge_types:
            [B, N, N]

        return:
            attn_bias [B * H, N, N]
        """
        bsz, n_node, _ = distances.shape

        gbf_feat = self.gbf(distances, edge_types)       # [B, N, N, 128]
        attn_bias = self.gbf_proj(gbf_feat)              # [B, N, N, 64]

        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()
        attn_bias = attn_bias.view(
            bsz * self.config.num_attention_heads,
            n_node,
            n_node,
        )

        return attn_bias

    def encode(
        self,
        tokens: torch.Tensor,
        distances: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_tokens(tokens)
        attn_bias = self.get_dist_features(distances, edge_types)

        encoder_output = self.encoder(
            x,
            attn_bias=attn_bias,
            padding_mask=padding_mask,
        )

        return encoder_output

    def forward(
        self,
        tokens: torch.Tensor,
        distances: torch.Tensor,
        edge_types: torch.Tensor,
        # labels: Optional[torch.Tensor] = None,
    ):
        padding_mask = tokens.eq(self.padding_idx)

        encoder_output = self.encode(
            tokens=tokens,
            distances=distances,
            edge_types=edge_types,
        )

        logits = self.prediction_head(
            encoder_output,
            padding_mask=padding_mask,
        )

        # loss = None

        # if labels is not None:
        #     if self.problem_type == "regression":
        #         labels = labels.float()
        #         if labels.ndim == 1:
        #             labels = labels.unsqueeze(-1)

        #         loss = F.mse_loss(logits.float(), labels)

        #     elif self.problem_type == "single_label_classification":
        #         loss = F.cross_entropy(
        #             logits.float(),
        #             labels.long(),
        #         )

        #     elif self.problem_type == "binary_classification":
        #         labels = labels.float()
        #         if labels.ndim == 1:
        #             labels = labels.unsqueeze(-1)

        #         loss = F.binary_cross_entropy_with_logits(
        #             logits.float(),
        #             labels,
        #         )

        #     elif self.problem_type == "multi_label_classification":
        #         loss = F.binary_cross_entropy_with_logits(
        #             logits.float(),
        #             labels.float(),
        #         )

        #     else:
        #         raise ValueError(f"Unknown problem_type: {self.problem_type}")

        # return {
        #     "loss": loss,
        #     "logits": logits,
        #     "encoder_output": encoder_output,
        # }
        
        return logits, encoder_output

    def load_unimol_pretrained(
        self,
        checkpoint_or_state_dict,
        strict: bool = False,
        verbose: bool = True,
    ):
        """
        checkpoint_or_state_dict:
            1) torch.load(...) 결과 전체
            2) checkpoint["model"]
            둘 다 가능.

        strict=False 권장.
        prediction_head.* 는 downstream task용 새 head라 checkpoint에 없는 게 정상.
        """

        if isinstance(checkpoint_or_state_dict, dict) and "model" in checkpoint_or_state_dict:
            state_dict = checkpoint_or_state_dict["model"]
        else:
            state_dict = checkpoint_or_state_dict

        result = self.load_state_dict(state_dict, strict=strict)

        if verbose:
            print("Missing keys:")
            for k in result.missing_keys:
                print("  ", k)

            print("Unexpected keys:")
            for k in result.unexpected_keys:
                print("  ", k)

        return result
