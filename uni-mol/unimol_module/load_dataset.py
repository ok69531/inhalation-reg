import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Uni-Mol molecule dictionary
# ============================================================

DEFAULT_UNIMOL_MOL_DICT = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "S": 7,
    "H": 8,
    "Cl": 9,
    "F": 10,
    "Br": 11,
    "I": 12,
    "Si": 13,
    "P": 14,
    "B": 15,
    "Na": 16,
    "K": 17,
    "Al": 18,
    "Ca": 19,
    "Sn": 20,
    "As": 21,
    "Hg": 22,
    "Fe": 23,
    "Zn": 24,
    "Cr": 25,
    "Se": 26,
    "Gd": 27,
    "Au": 28,
    "Li": 29,
    "[MASK]": 30,
}


def load_unimol_dictionary(path, add_mask = True) -> Dict[str, int]:
    """
    Uni-Mol dict 파일을 token -> id dictionary로 읽는다.

    dict 파일 형식이 아래 둘 중 하나여도 처리함.

    예시 1:
        C
        N
        O

    예시 2:
        C 12345
        N 67890
        O 11111
    """
    path = Path(path)

    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            tokens.append(token)

    special = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]

    if not all(s in tokens for s in special):
        tokens = special + [t for t in tokens if t not in special]

    if add_mask and "[MASK]" not in tokens:
        tokens.append("[MASK]")

    dictionary = {tok: idx for idx, tok in enumerate(tokens)}
    return dictionary


# ============================================================
# 2. SMILES validation
# ============================================================

def filter_valid_smiles_rows(df, smiles_col) -> pd.DataFrame:
    """
    Chem.MolFromSmiles(smiles)가 None인 행 제거.

    canonicalize=True면 valid SMILES를 canonical SMILES로 바꿔 저장.
    """
    from rdkit import Chem

    valid_indices = []
    valid_smiles = []

    for idx, smi in df[smiles_col].items():
        if pd.isna(smi):
            continue

        smi = str(smi).strip()
        if smi == "":
            continue

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        valid_indices.append(idx)
        valid_smiles.append(smi)

    filtered = df.loc[valid_indices].copy()
    filtered[smiles_col] = valid_smiles
    filtered = filtered.reset_index(drop=True)

    return filtered


# ============================================================
# 3. SMILES -> atoms, coords
# ============================================================

def smiles_to_atoms_coords(
    smiles: str,
    remove_hs: bool = False,
    seed: int = 0,
    optimize: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    SMILES에서 RDKit 3D conformer를 생성하고 atoms, coords를 반환.

    remove_hs=True:
        Uni-Mol mol_pre_no_h checkpoint 사용 시 보통 True.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxAttempts = 1000

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None

    if optimize:
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props is not None:
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()

    atoms = []
    coords = []

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = conf.GetAtomPosition(atom_idx)

        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])

    coords = np.asarray(coords, dtype=np.float32)

    return {
        "atoms": atoms,
        "coords": coords,
    }


# ============================================================
# 4. atoms, coords -> Uni-Mol input
# ============================================================

def atoms_coords_to_unimol_input(
    atoms: List[str],
    coords: Union[np.ndarray, torch.Tensor],
    dictionary: Optional[Dict[str, int]] = None,
    max_atoms: int = 256,
    center_coords: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    atoms, coords를 Uni-Mol 입력으로 변환.

    출력:
        tokens: LongTensor [N]
        coords: FloatTensor [N, 3]
        distances: FloatTensor [N, N]
        edge_types: LongTensor [N, N]

    N = 실제 원자 수 + 2
    앞에 [CLS], 뒤에 [SEP]를 붙인다.
    """

    if dictionary is None:
        dictionary = DEFAULT_UNIMOL_MOL_DICT

    cls_idx = dictionary["[CLS]"]
    sep_idx = dictionary["[SEP]"]
    unk_idx = dictionary["[UNK]"]

    vocab_size = len(dictionary)

    if isinstance(coords, np.ndarray):
        coords = torch.tensor(coords, dtype=torch.float32)
    else:
        coords = coords.float()

    if len(atoms) != coords.shape[0]:
        raise ValueError(
            f"atoms 길이와 coords 길이가 다름: len(atoms)={len(atoms)}, coords.shape={coords.shape}"
        )

    max_real_atoms = max_atoms - 2

    if len(atoms) > max_real_atoms:
        atoms = atoms[:max_real_atoms]
        coords = coords[:max_real_atoms]

    if center_coords and coords.numel() > 0:
        coords = coords - coords.mean(dim=0, keepdim=True)

    token_ids = [cls_idx]
    token_ids += [dictionary.get(atom, unk_idx) for atom in atoms]
    token_ids += [sep_idx]

    tokens = torch.tensor(token_ids, dtype=torch.long)

    zero = torch.zeros(1, 3, dtype=torch.float32)

    coords_with_special = torch.cat(
        [
            zero,
            coords.float(),
            zero,
        ],
        dim=0,
    )

    distances = torch.cdist(coords_with_special, coords_with_special)

    edge_types = (
        tokens.view(-1, 1) * vocab_size
        + tokens.view(1, -1)
    ).long()

    return {
        "tokens": tokens,
        "coords": coords_with_special,
        "distances": distances,
        "edge_types": edge_types,
    }


def smiles_to_unimol_input(
    smiles: str,
    dictionary: Optional[Dict[str, int]] = None,
    max_atoms: int = 256,
    remove_hs: bool = False,
    center_coords: bool = True,
    seed: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    SMILES 하나를 Uni-Mol 입력으로 변환.
    conformer 생성 실패 시 None 반환.
    """
    mol_data = smiles_to_atoms_coords(
        smiles=smiles,
        remove_hs=remove_hs,
        seed=seed,
    )

    if mol_data is None:
        return None

    return atoms_coords_to_unimol_input(
        atoms=mol_data["atoms"],
        coords=mol_data["coords"],
        dictionary=dictionary,
        max_atoms=max_atoms,
        center_coords=center_coords,
    )


# ============================================================
# 5. label 처리
# ============================================================

def extract_label(
    row: pd.Series,
    label_cols: Union[str, List[str]],
    problem_type: str,
) -> torch.Tensor:
    """
    label_cols:
        "target" 또는 ["target1", "target2"]

    problem_type:
        regression
        single_label_classification
        binary_classification
        multi_label_classification
    """
    if isinstance(label_cols, str):
        value = row[label_cols]

        if problem_type == "single_label_classification":
            return torch.tensor(int(value), dtype=torch.long)

        return torch.tensor(float(value), dtype=torch.float32)

    values = [row[c] for c in label_cols]

    if problem_type == "single_label_classification":
        if len(values) != 1:
            raise ValueError("single_label_classification은 label column 하나만 사용해야 함.")
        return torch.tensor(int(values[0]), dtype=torch.long)

    return torch.tensor(values, dtype=torch.float32)


# ============================================================
# 6. Excel -> train/test preprocessing
# ============================================================

def preprocess_excel_to_unimol_samples(
    excel_path: Union[str, Path],
    smiles_col: str,
    label_cols: Union[str, List[str]],
    dictionary: Optional[Dict[str, int]] = None,
    sheet_name: Union[str, int] = 0,
    test_size: float = 0.2,
    random_state: int = 42,
    problem_type: str = "regression",
    max_atoms: int = 256,
    remove_hs: bool = False,
    center_coords: bool = True,
) -> Dict[str, Any]:
    """
    Excel 파일을 읽고 Uni-Mol용 train/test samples로 전처리.

    처리 순서:
        1. Excel 읽기
        2. smiles_col / label_cols 결측 제거
        3. Chem.MolFromSmiles가 None인 행 제거
        4. train_test_split
        5. 각 SMILES를 tokens, distances, edge_types로 변환
    """

    if dictionary is None:
        dictionary = DEFAULT_UNIMOL_MOL_DICT

    excel_path = Path(excel_path)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    needed_cols = [smiles_col]
    if isinstance(label_cols, str):
        needed_cols.append(label_cols)
    else:
        needed_cols.extend(label_cols)

    df = df.dropna(subset=needed_cols).copy()
    df = df.reset_index(drop=True)

    before_valid = len(df)
    df = filter_valid_smiles_rows(
        df,
        smiles_col=smiles_col,
    )
    after_valid = len(df)

    print(f"[SMILES filter] before={before_valid}, valid={after_valid}, removed={before_valid - after_valid}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        # stratify=stratify_values,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Split] train={len(train_df)}, test={len(test_df)}")

    train_samples, train_failed = dataframe_to_unimol_samples(
        train_df,
        smiles_col=smiles_col,
        label_cols=label_cols,
        dictionary=dictionary,
        problem_type=problem_type,
        max_atoms=max_atoms,
        remove_hs=remove_hs,
        center_coords=center_coords,
        seed_offset=0,
    )

    test_samples, test_failed = dataframe_to_unimol_samples(
        test_df,
        smiles_col=smiles_col,
        label_cols=label_cols,
        dictionary=dictionary,
        problem_type=problem_type,
        max_atoms=max_atoms,
        remove_hs=remove_hs,
        center_coords=center_coords,
        seed_offset=100000,
    )

    print(f"[Conformer] train_ok={len(train_samples)}, train_failed={train_failed}")
    print(f"[Conformer] test_ok={len(test_samples)}, test_failed={test_failed}")

    return {
        "train": train_samples,
        "test": test_samples,
        "meta": {
            "excel_path": str(excel_path),
            "smiles_col": smiles_col,
            "label_cols": label_cols,
            "problem_type": problem_type,
            "test_size": test_size,
            "random_state": random_state,
            "max_atoms": max_atoms,
            "remove_hs": remove_hs,
            "center_coords": center_coords,
            "vocab_size": len(dictionary),
            "num_train": len(train_samples),
            "num_test": len(test_samples),
            "num_invalid_smiles_removed": before_valid - after_valid,
            "num_train_conformer_failed": train_failed,
            "num_test_conformer_failed": test_failed,
        },
    }


def dataframe_to_unimol_samples(
    df: pd.DataFrame,
    smiles_col: str,
    label_cols: Union[str, List[str]],
    dictionary: Dict[str, int],
    problem_type: str,
    max_atoms: int,
    remove_hs: bool,
    center_coords: bool,
    seed_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    DataFrame을 Uni-Mol sample list로 변환.
    conformer 생성 실패한 sample은 제외.
    """
    samples = []
    failed = 0

    for i, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()

        sample = smiles_to_unimol_input(
            smiles=smiles,
            dictionary=dictionary,
            max_atoms=max_atoms,
            remove_hs=remove_hs,
            center_coords=center_coords,
            seed=seed_offset + i,
        )

        if sample is None:
            failed += 1
            continue

        label = extract_label(
            row=row,
            label_cols=label_cols,
            problem_type=problem_type,
        )

        sample["label"] = label
        sample["smiles"] = smiles

        samples.append(sample)

    return samples, failed


# ============================================================
# 7. cache 저장 / 로드
# ============================================================

def torch_load_compat(path: Union[str, Path]):
    """
    PyTorch 버전 차이 대응용 torch.load.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_or_preprocess_unimol_excel(
    tg_num: int,
    excel_path: Union[str, Path],
    cache_path: Union[str, Path],
    smiles_col: str,
    label_cols: Union[str, List[str]],
    dictionary: Optional[Dict[str, int]] = None,
    sheet_name: Union[str, int] = 0,
    test_size: float = 0.2,
    random_state: int = 42,
    problem_type: str = "regression",
    max_atoms: int = 256,
    remove_hs: bool = False,
    center_coords: bool = True,
) -> Dict[str, Any]:
    """
    저장 파일이 있으면 torch.load로 불러오고,
    없으면 Excel에서 전처리 후 torch.save.

    반환:
        {
            "train": List[Dict],
            "test": List[Dict],
            "meta": Dict
        }
    """
    cache_path = os.path.join(cache_path, f'tg{tg_num}.pt')
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
    else:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if tg_num == 403:
            excel_path = os.path.join(excel_path, f'tg{tg_num}.xlsx')
        elif tg_num == 412:
            excel_path = os.path.join(excel_path, f'tg{tg_num}_413.xlsx')
        
        data = preprocess_excel_to_unimol_samples(
            excel_path=excel_path,
            smiles_col=smiles_col,
            label_cols=label_cols,
            dictionary=dictionary,
            sheet_name=sheet_name,
            test_size=test_size,
            random_state=random_state,
            problem_type=problem_type,
            max_atoms=max_atoms,
            remove_hs=remove_hs,
            center_coords=center_coords,
        )

        torch.save(data, cache_path)
        print(f"[Cache] saved: {cache_path}")

    return data


# ============================================================
# 8. Dataset / collate_fn
# ============================================================

class UniMolPropertyDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def unimol_collate_fn(
    samples: List[Dict[str, Any]],
    pad_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    서로 길이가 다른 molecule들을 padding해서 batch로 묶는다.

    입력 sample:
        tokens: [N]
        distances: [N, N]
        edge_types: [N, N]
        label: scalar or vector

    출력 batch:
        tokens: [B, Nmax]
        distances: [B, Nmax, Nmax]
        edge_types: [B, Nmax, Nmax]
        labels: [B] or [B, C]
    """
    max_len = max(s["tokens"].shape[0] for s in samples)

    batch_tokens = []
    batch_distances = []
    batch_edge_types = []
    batch_labels = []

    for s in samples:
        tokens = s["tokens"].long()
        distances = s["distances"].float()
        edge_types = s["edge_types"].long()
        label = s["label"]

        n = tokens.shape[0]
        pad_len = max_len - n

        tokens = F.pad(
            tokens,
            pad=(0, pad_len),
            value=pad_idx,
        )

        distances = F.pad(
            distances,
            pad=(0, pad_len, 0, pad_len),
            value=0.0,
        )

        edge_types = F.pad(
            edge_types,
            pad=(0, pad_len, 0, pad_len),
            value=0,
        )

        batch_tokens.append(tokens)
        batch_distances.append(distances)
        batch_edge_types.append(edge_types)
        batch_labels.append(torch.as_tensor(label))

    return {
        "tokens": torch.stack(batch_tokens, dim=0),
        "distances": torch.stack(batch_distances, dim=0),
        "edge_types": torch.stack(batch_edge_types, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
    }


# ============================================================
# 9. 사용 예시
# ============================================================

if __name__ == "__main__":
    # 여기를 네 파일/컬럼 이름에 맞게 수정
    EXCEL_PATH = "../../dataset/raw"
    CACHE_PATH = "dataset/processed"

    SMILES_COL = "smiles"
    LABEL_COLS = "value"

    # regression:
    #   problem_type="regression"
    #
    # binary classification:
    #   problem_type="binary_classification"
    #
    # multiclass classification:
    #   problem_type="single_label_classification"
    #
    # multilabel classification:
    #   problem_type="multi_label_classification"
    PROBLEM_TYPE = "regression"

    dictionary = DEFAULT_UNIMOL_MOL_DICT

    data = load_or_preprocess_unimol_excel(
        tg_num=403,
        excel_path=EXCEL_PATH,
        cache_path=CACHE_PATH,
        smiles_col=SMILES_COL,
        label_cols=LABEL_COLS,
        dictionary=dictionary,
    )

    train_dataset = UniMolPropertyDataset(data["train"])
    test_dataset = UniMolPropertyDataset(data["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=unimol_collate_fn,
    )
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=16,
    #     shuffle=False,
    #     collate_fn=unimol_collate_fn,
    # )

    # batch = next(iter(train_loader))

    # print(batch["tokens"].shape)
    # print(batch["distances"].shape)
    # print(batch["edge_types"].shape)
    # print(batch["labels"].shape)

    # 모델에 넣을 때:
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    #
    # for batch in train_loader:
    #     tokens = batch["tokens"].to(device)
    #     distances = batch["distances"].to(device)
    #     edge_types = batch["edge_types"].to(device)
    #     labels = batch["labels"].to(device)
    #
    #     outputs = model(
    #         tokens=tokens,
    #         distances=distances,
    #         edge_types=edge_types,
    #         labels=labels,
    #     )
    #
    #     loss = outputs["loss"]
    #     loss.backward()