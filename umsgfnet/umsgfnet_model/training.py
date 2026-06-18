
# ===== Case analysis runner (drop-in) =====



import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from UMSGFNet_lstm import UMSGFNet
from splitters import scaffold_split
from loader import HiMolGraph, MoleculeDataset
from explain import (
    compute_atom_importance_occlusion, draw_atom_importance,
    compute_fp_group_importance, draw_fp_group_heatmap
)
import re

from explain import compute_atom_importance_occlusion, draw_atom_importance

#from explain import compute_atom_importance, draw_atom_importance

criterion = nn.BCEWithLogitsLoss(reduction="none")


def _sanitize_filename(name: str, fallback: str = "mol"):
    """把任意 mol_id 清洗成安全文件名：仅保留 A-Za-z0-9._-，其他替换为 '_'。"""
    name = (str(name) if name is not None else "").strip().replace(" ", "_")
    safe = "_".join(re.findall(r"[A-Za-z0-9._-]+", name))
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or fallback

def _extract_ids_from_batch(batch):
    """
    尝试从 batch 中抽取每个样本的 ID 列表：
    优先使用 batch.mol_id / batch.molids / ...，否则用 batch.id。
    返回 ['id0','id1', ...]；找不到就返回 None。
    """
    cand = ["mol_id", "molids", "molIds", "ids", "id", "MOL_ID", "MolID"]
    field = None
    for n in cand:
        if hasattr(batch, n):
            field = getattr(batch, n)
            break
    if field is None:
        return None

    def _tolist(x):
        if isinstance(x, (list, tuple)): return list(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if torch.is_tensor(x): return x.detach().cpu().tolist()
        return [x]

    raw = _tolist(field)
    out = []
    for v in raw:
        if isinstance(v, (bytes, bytearray)):
            try: v = v.decode("utf-8")
            except Exception: v = str(v)
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if v:
            out.append(v)
    return out or None







def train(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()


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
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        huber_loss = nn.SmoothL1Loss(beta=args.huber_beta)
        loss = huber_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader, save_path):
    model.eval()
    y_true = []
    y_scores = []
    cids = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        cids.append(batch.id.cpu().numpy())

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    cids = np.concatenate(cids, axis=0)

    if save_path:
        df = pd.DataFrame({
            "CID": cids,
            "y_true": y_true.flatten(),
            "y_pred": y_scores.flatten()
        })
        df.to_csv(save_path, index=False)

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    eval_roc = (sum(roc_list) / len(roc_list)) if len(roc_list) > 0 else float("nan")
    loss = float("nan")
    return eval_roc, loss


def eval_reg(args, model, device, loader, save_path):
    model.eval()
    y_true = []
    y_scores = []
    cids = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        cids.append(batch.id.cpu().numpy())

    y_true = torch.cat(y_true, dim=0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy().flatten()
    cids = np.concatenate(cids, axis=0)

    if save_path:
        df = pd.DataFrame({
            "CID": cids,
            "y_true": y_true,
            "y_pred": y_scores
        })
        df.to_csv(save_path, index=False)

    # --- 修改开始 ---
    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_scores)  # 新增 R2 计算

    return mse, mae, rmse, r2  # 返回 4 个值
    # --- 修改结束 ---


def run_case_analysis(model, device, example_batch, args, epoch):
    """
    在训练过程中运行案例分析：
      - 原子级遮挡可视化
      - 指纹组级遮挡热力图
    """
    import numpy as np, os

    os.makedirs(args.case_dir, exist_ok=True)

    # === 用 case_index 选一个样本，并取它的 mol_id（或 id）作为文件名前缀 ===
    idx = int(getattr(args, "case_index", 0))
    ids = _extract_ids_from_batch(example_batch)  # 可能是 batch.mol_id 或 batch.id
    if ids is not None and len(ids) > 0:
        raw_id = ids[idx] if idx < len(ids) else ids[0]
    else:
        raw_id = f"mol_{idx}"
    stem = _sanitize_filename(raw_id, fallback=f"mol_{idx}")

    # ----------------- 原子级解释 -----------------
    try:
        smi, scores, raw_scores, pairs = compute_atom_importance_occlusion(
            model, example_batch, device,
            target_index=0,
            cls_strategy=("sum" if args.dataset.lower()=="esol" else "auto")
        )
        out_png = os.path.join(args.case_dir, f"{stem}_epoch{epoch:03d}_atom.png")
        draw_atom_importance(smi, scores, out_png)

        np.savetxt(os.path.join(args.case_dir, f"{stem}_epoch{epoch:03d}_atom_raw.csv"),
                   np.array(raw_scores, dtype=np.float32), delimiter=",")
        np.savetxt(os.path.join(args.case_dir, f"{stem}_epoch{epoch:03d}_atom_norm.csv"),
                   np.array(scores, dtype=np.float32), delimiter=",")
    except Exception as e:
        print(f"[Warn] Atom-level case analysis failed at epoch {epoch}: {e}")

    # ----------------- 指纹组级解释 -----------------
    try:
        g_norm, g_raw, slices = compute_fp_group_importance(
            model, example_batch, device,
            target_index=0,
            cls_strategy=("sum" if args.dataset.lower()=="esol" else "auto")
        )
        group_names = [n for n,_,_ in slices]

        raw_vec  = np.array([g_raw.get(n, 0.0)  for n in group_names], dtype=np.float32)
        norm_vec = np.array([g_norm.get(n, 0.0) for n in group_names], dtype=np.float32)

        np.savetxt(os.path.join(args.case_dir, f"{args.dataset}_epoch{epoch:03d}_fp_raw.csv"),
                   raw_vec, delimiter=",", header=",".join(group_names), comments="")
        np.savetxt(os.path.join(args.case_dir, f"{args.dataset}_epoch{epoch:03d}_fp_norm.csv"),
                   norm_vec, delimiter=",", header=",".join(group_names), comments="")

        heatmap_png = os.path.join(args.case_dir, f"{args.dataset}_epoch{epoch:03d}_fp_heatmap.png")
        draw_fp_group_heatmap(
            heatmap_png,
            group_scores_list=[g_norm],   # 单个 epoch 的字典，包成 list
            group_order=group_names,
            dpi=400,
            title=f"{args.dataset.upper()} fingerprint-group contributions (epoch {epoch})"
        )
    except Exception as e:
        print(f"[Warn] FP-group case analysis failed at epoch {epoch}: {e}")



def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of UMSGFNet with per-epoch case analysis')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001 , help='learning rate for the prediction layer')
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='[bbbp, bace, sider, clintox,tox21, toxcast, esol,freesolv,lipophilicity]')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help="the path of input CSV file")
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints', help="the path to save output model")
    parser.add_argument('--depth', type=int, default=7, help="the depth of molecule encoder")
    parser.add_argument('--seed', type=int, default=88, help="seed for splitting the dataset")
    parser.add_argument('--runseed', type=int, default=88, help="seed for minibatch selection, random initialization")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--huber_beta', type=float, default=7.0, help='Beta parameter for Huber loss')
    parser.add_argument('--weight_power', type=float, default=1.0, help='Power for error-based weighting in Weighted Huber Loss')
    parser.add_argument('--weight_epsilon', type=float, default=1e-6, help='Epsilon for error-based weighting in Weighted Huber Loss')
    parser.add_argument('--example_index', type=int, default=0, help='每个 epoch 进行案例分析的 test 集样本索引')
    #parser.add_argument('--case_dir', type=str, default='./case_analysis', help='保存案例图的目录')
    parser.add_argument("--case_dir", type=str, default="case_outputs",
                        help="目录：保存案例分析图片与CSV")
    parser.add_argument("--case_every", type=int, default=1,
                        help="每隔多少个epoch做一次案例分析")
    parser.add_argument("--case_index", type=int, default=0,
                        help="从batch中选第几个分子做案例分析（0为第一个）")
    parser.add_argument("--dataset_name", type=str, default="tox21",
                        help="用于命名 & 选择解释策略（esol=回归）")

    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'bace', 'bbbp', 'sider', 'clintox']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    print('process data')
    dataset = MoleculeDataset(os.path.join(args.data_dir, args.dataset), dataset=args.dataset)

    print("scaffold")
    smiles_list = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'processed', 'smiles.csv'),
                              header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0,
                                                                frac_train=0.8, frac_valid=0.1,
                                                                frac_test=0.1, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UMSGFNet(data_name=args.dataset, atom_fdim=89, bond_fdim=98, fp_fdim=6338,
                  hidden_size=512, depth=args.depth, device=device, out_dim=num_tasks)
    model.to(device)

    model_param_group = []
    model_param_group.append({"params": model.parameters(), "lr": args.lr})
    optimizer = optim.Adam(model_param_group, weight_decay=1e-3)
    print(optimizer)

    os.makedirs(args.case_dir, exist_ok=True)

    fixed_example_batch = None
    for _b in train_loader:
        fixed_example_batch = _b
        break
    if fixed_example_batch is None:
        raise RuntimeError("训练集为空，无法进行案例分析")

    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, args.dataset + '.pth')

    # example_index = max(0, min(args.example_index, len(test_dataset) - 1))
    # example_data = test_dataset[example_index]

    if task_type == 'cls':
        best_auc = 0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            print('====epoch:', epoch)
            train(model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_auc, _ = eval(args, model, device, train_loader, save_path=None)
            else:
                train_auc = 0
            val_auc, _ = eval(args, model, device, val_loader, save_path=None)
            test_auc, _ = eval(args, model, device, test_loader, save_path=None)

            if test_auc > best_auc or np.isnan(best_auc):
                best_auc = test_auc
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)

            print(f"train_auc: {train_auc:.4f} val_auc: {val_auc:.4f} test_auc: {test_auc:.4f}  (best@{best_epoch}={best_auc:.4f})")
            if (epoch % args.case_every) == 0:
                run_case_analysis(model, device, fixed_example_batch, args, epoch)


            # try:
            #
            #     # 计算（规范化分数、原始分数、及逐原子的(base, ablated)）
            #     smi, scores_norm, scores_raw, scalar_pairs = compute_atom_importance_occlusion(
            #         model, example_data, device, target_index=0, cls_strategy="auto"  # 回归任务可用 "sum"
            #     )
            #
            #     # 1) 保存热力图（用规范化分数）
            #     out_png = os.path.join(args.case_dir, f'epoch_{epoch:03d}.png')
            #     draw_atom_importance(smi, scores_norm, out_png)
            #
            #     # 2) 保存规范化分数（用于可视化复现）
            #     np.savetxt(
            #         os.path.join(args.case_dir, f'epoch_{epoch:03d}_scores.csv'),
            #         np.array(scores_norm, dtype=np.float32),
            #         delimiter=','
            #     )
            #
            #     # 3) 保存原始差值分数（跨 epoch 直接可比）
            #     np.savetxt(
            #         os.path.join(args.case_dir, f'epoch_{epoch:03d}_raw_scores.csv'),
            #         np.array(scores_raw, dtype=np.float32),
            #         delimiter=','
            #     )
            #
            #     # 4) 保存每个原子的 (base, ablated) 标量，便于检查遮挡前后预测值
            #     base_abl = np.array(scalar_pairs, dtype=np.float32)  # shape [n_atoms, 2]
            #     np.savetxt(
            #         os.path.join(args.case_dir, f'epoch_{epoch:03d}_scalar_pairs.csv'),
            #         base_abl,
            #         delimiter=',',
            #         header='base,ablated',
            #         comments=''
            #     )
            #
            # except Exception as e:
            #     print(f"[Warn] Case analysis failed at epoch {epoch}: {e}")




    elif task_type == 'reg':
        best_rmse = float("inf")
        for epoch in range(1, args.epochs + 1):
            print('====epoch:', epoch)
            train_reg(args, model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                # --- 修改：接收 4 个返回值 ---
                train_mse, train_mae, train_rmse, train_r2 = eval_reg(args, model, device, train_loader, save_path=None)
            else:
                train_mse, train_mae, train_rmse, train_r2 = 0, 0, 0, 0

                # --- 修改：接收 4 个返回值 ---
            val_mse, val_mae, val_rmse, val_r2 = eval_reg(args, model, device, val_loader, save_path=None)
            test_mse, test_mae, test_rmse, test_r2 = eval_reg(args, model, device, test_loader, save_path=None)

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save(model.state_dict(), model_save_path)

                # --- 修改：打印所有指标 ---
                print(f"Train - RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R2: {train_r2:.4f}")
                print(f"Val   - RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f}")
                print(f"Test  - RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")
                print(f"Best Test RMSE so far: {best_rmse:.4f}")
            if (epoch % args.case_every) == 0:
                run_case_analysis(model, device, fixed_example_batch, args, epoch)

            # # ---- 每个 epoch 做一次案例分析并保存图 ----
            # try:
            #     # 回归：用 "sum" 取标量
            #     smi, scores_norm, scores_raw, scalar_pairs = compute_atom_importance_occlusion(
            #         model, example_data, device, target_index=0, cls_strategy="sum"
            #     )
            #     # 图像（用规范化后的分数绘图）
            #     out_png = os.path.join(args.case_dir, 'epoch_{:03d}.png'.format(epoch))
            #     draw_atom_importance(smi, scores_norm, out_png)
            #
            #     # 保存数值
            #     np.savetxt(os.path.join(args.case_dir, 'epoch_{:03d}_scores.csv'.format(epoch)),
            #                np.array(scores_norm, dtype=np.float32), delimiter=',')
            #     np.savetxt(os.path.join(args.case_dir, 'epoch_{:03d}_raw_scores.csv'.format(epoch)),
            #                np.array(scores_raw, dtype=np.float32), delimiter=',')
            #
            #     # 每原子的 (base, ablated) 标量，有助于诊断
            #     base_abl = np.array(scalar_pairs, dtype=np.float32)  # shape [n_atoms, 2]
            #     np.savetxt(os.path.join(args.case_dir, 'epoch_{:03d}_scalar_pairs.csv'.format(epoch)),
            #                base_abl, delimiter=',', header='base,ablated', comments='')
            # except Exception as e:
            #     print(f"[Warn] Case analysis failed at epoch {epoch}: {e}")


if __name__ == "__main__":
    main()

