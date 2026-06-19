
# ===== Case analysis runner (drop-in) =====
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# from explain import (
#     compute_atom_importance_occlusion, draw_atom_importance,
#     compute_fp_group_importance, draw_fp_group_heatmap
# )


# criterion = nn.BCEWithLogitsLoss(reduction="none")


class WeightedHuberLoss(nn.Module):
    def __init__(self, beta: float = 1.0, power: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        self.huber_loss = nn.SmoothL1Loss(reduction='none', beta=beta)
        self.power = power
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.abs(target - pred)
        weights = error ** self.power + self.epsilon
        huber_individual_losses = self.huber_loss(pred, target)
        weighted_huber_loss = weights * huber_individual_losses
        return torch.mean(weighted_huber_loss)


def train_reg(args, model, device, loader, optimizer):
    model.train()
    
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch, 'train')
        y = batch.y.view(pred.shape).to(torch.float64)
        huber_loss = nn.SmoothL1Loss(beta=args.huber_beta)
        loss = huber_loss(pred, y)
        total_loss += loss.item() * len(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_reg(model, device, loader, split_tag):
    model.eval()
    
    y, pred = [], []
    origin_y, origin_pred = [], []

    for _, batch in enumerate(loader):
        batch = batch.to(device)
        scores = model(batch, split_tag)
        
        y.append(batch.y)
        pred.append(scores.view(-1))
        
        origin_y.append(batch.origin_y)
        origin_pred.append(10 ** scores.view(-1))

    y = torch.cat(y).cpu().numpy()
    pred = torch.cat(pred, dim = 0).cpu().detach().numpy()
    origin_y = torch.cat(origin_y).cpu().numpy()
    origin_pred = torch.cat(origin_pred, dim = 0).cpu().detach().numpy()
    # cids = np.concatenate(cids, axis=0)

    # if save_path:
    #     df = pd.DataFrame({
    #         "CID": cids,
    #         "y_true": y_true,
    #         "y_pred": y_scores
    #     })
    #     df.to_csv(save_path, index=False)

    graph_metric = {
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

    return graph_metric['log_mae'], graph_metric, pred_result




# def _sanitize_filename(name: str, fallback: str = "mol"):
#     name = (str(name) if name is not None else "").strip().replace(" ", "_")
#     safe = "_".join(re.findall(r"[A-Za-z0-9._-]+", name))
#     safe = re.sub(r"_+", "_", safe).strip("._-")
#     return safe or fallback


# def _extract_ids_from_batch(batch):
#     """
#     尝试从 batch 中抽取每个样本的 ID 列表：
#     优先使用 batch.mol_id / batch.molids / ...，否则用 batch.id。
#     返回 ['id0','id1', ...]；找不到就返回 None。
#     """
#     cand = ["mol_id", "molids", "molIds", "ids", "id", "MOL_ID", "MolID"]
#     field = None
#     for n in cand:
#         if hasattr(batch, n):
#             field = getattr(batch, n)
#             break
#     if field is None:
#         return None

#     def _tolist(x):
#         if isinstance(x, (list, tuple)): return list(x)
#         if isinstance(x, np.ndarray): return x.tolist()
#         if torch.is_tensor(x): return x.detach().cpu().tolist()
#         return [x]

#     raw = _tolist(field)
#     out = []
#     for v in raw:
#         if isinstance(v, (bytes, bytearray)):
#             try: v = v.decode("utf-8")
#             except Exception: v = str(v)
#         if not isinstance(v, str):
#             v = str(v)
#         v = v.strip()
#         if v:
#             out.append(v)
#     return out or None


# def run_case_analysis(model, device, example_batch, args, epoch):
#     """
#     在训练过程中运行案例分析：
#       - 原子级遮挡可视化
#       - 指纹组级遮挡热力图
#     """
#     import numpy as np, os

#     os.makedirs(args.case_dir, exist_ok=True)

#     # === 用 case_index 选一个样本，并取它的 mol_id（或 id）作为文件名前缀 ===
#     idx = int(getattr(args, "case_index", 0))
#     ids = _extract_ids_from_batch(example_batch)  # 可能是 batch.mol_id 或 batch.id
#     if ids is not None and len(ids) > 0:
#         raw_id = ids[idx] if idx < len(ids) else ids[0]
#     else:
#         raw_id = f"mol_{idx}"
#     stem = _sanitize_filename(raw_id, fallback=f"mol_{idx}")

#     # ----------------- 原子级解释 -----------------
#     try:
#         smi, scores, raw_scores, pairs = compute_atom_importance_occlusion(
#             model, example_batch, device,
#             target_index=0,
#             cls_strategy=("sum" if args.dataset.lower()=="esol" else "auto")
#         )
#         out_png = os.path.join(args.case_dir, f"{stem}_epoch{epoch:03d}_atom.png")
#         draw_atom_importance(smi, scores, out_png)

#         np.savetxt(os.path.join(args.case_dir, f"{stem}_epoch{epoch:03d}_atom_raw.csv"),
#                    np.array(raw_scores, dtype=np.float32), delimiter=",")
#         np.savetxt(os.path.join(args.case_dir, f"{stem}_epoch{epoch:03d}_atom_norm.csv"),
#                    np.array(scores, dtype=np.float32), delimiter=",")
#     except Exception as e:
#         print(f"[Warn] Atom-level case analysis failed at epoch {epoch}: {e}")

#     # ----------------- 指纹组级解释 -----------------
#     try:
#         g_norm, g_raw, slices = compute_fp_group_importance(
#             model, example_batch, device,
#             target_index=0,
#             cls_strategy=("sum" if args.dataset.lower()=="esol" else "auto")
#         )
#         group_names = [n for n,_,_ in slices]

#         raw_vec  = np.array([g_raw.get(n, 0.0)  for n in group_names], dtype=np.float32)
#         norm_vec = np.array([g_norm.get(n, 0.0) for n in group_names], dtype=np.float32)

#         np.savetxt(os.path.join(args.case_dir, f"{args.dataset}_epoch{epoch:03d}_fp_raw.csv"),
#                    raw_vec, delimiter=",", header=",".join(group_names), comments="")
#         np.savetxt(os.path.join(args.case_dir, f"{args.dataset}_epoch{epoch:03d}_fp_norm.csv"),
#                    norm_vec, delimiter=",", header=",".join(group_names), comments="")

#         heatmap_png = os.path.join(args.case_dir, f"{args.dataset}_epoch{epoch:03d}_fp_heatmap.png")
#         draw_fp_group_heatmap(
#             heatmap_png,
#             group_scores_list=[g_norm],   # 单个 epoch 的字典，包成 list
#             group_order=group_names,
#             dpi=400,
#             title=f"{args.dataset.upper()} fingerprint-group contributions (epoch {epoch})"
#         )
#     except Exception as e:
#         print(f"[Warn] FP-group case analysis failed at epoch {epoch}: {e}")

