import os
import pickle

from typing import List, Tuple, Union

import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors


fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


# ATOM_FDIM = 91
BOND_FDIM = 9
FEATURES = {
    'atomic_num': [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 53,
                   56, 57, 58, 60, 62, 63, 64, 66, 70, 74, 78, 79, 80, 81, 82, 83, 88, 98],
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-1, -2, 1, 2, 3, 0],
    'chiral_tag': [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2],
    'stereo': [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREOE], }
ATOM_FDIM = (
    len(FEATURES["atomic_num"])
    + len(FEATURES["degree"])
    + len(FEATURES["formal_charge"])
    + len(FEATURES["chiral_tag"])
    + len(FEATURES["num_Hs"])
    + len(FEATURES["hybridization"])
    + 1  # aromatic
)

def get_atom_fdim() -> int:
    return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    encoding = [0] * len(choices)
    index = choices.index(value)
    encoding[index] = 1
    return encoding


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    if bond is None:
        fbond = [0] * BOND_FDIM
    else:
        bt = bond.GetBondType()
        fbond = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(bond.GetStereo(), FEATURES['stereo'])
    return fbond


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    features = onek_encoding_unk(atom.GetAtomicNum(), FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0]
    if functional_groups is not None:
        features += functional_groups
    return features


def pharm_feats(mol, factory=factory):
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


def get_fp_feature(mol):
    try:
        fp_atomPairs = list(rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=2048, use2D=True))
        fp_maccs = list(MACCSkeys.GenMACCSKeys(mol))
        fp_morganBits = list(GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS))
        fp_morganCounts = list(AllChem.GetHashedMorganFingerprint(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS))
        fp_pharm = pharm_feats(mol)
    except Exception:
        fp_atomPairs = [0 for _ in range(2048)]
        fp_maccs = [0 for _ in range(167)]
        fp_morganBits = [0 for _ in range(2048)]
        fp_morganCounts = [0 for _ in range(2048)]
        fp_pharm = [0 for _ in range(27)]
    return fp_atomPairs + fp_maccs + fp_morganBits + fp_morganCounts + fp_pharm


def get_cliques_link(breaks, cliques):
    clique_id_of_node = {}
    for idx, clique in enumerate(cliques):
        for c in clique:
            clique_id_of_node[c] = idx
    breaks_bond = {}
    for bond in breaks:
        breaks_bond[f'{bond[0]}_{bond[1]}'] = [clique_id_of_node[bond[0]], clique_id_of_node[bond[1]]]
    return breaks_bond


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])
            breaks.append([bond[0][0], bond[0][1]])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    breaks = get_cliques_link(breaks, cliques)
    return breaks, cliques


class BatchMolGraph:
    def __init__(self, smiles, atom_fdim, bond_fdim, fp_fdim, tg_num, split_tag):
        import ast, re

        def _maybe_strip_bytes_literal(s):
            # 处理 b'CCO' / "b'CCO'"
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

        def _maybe_parse_list_string(s):
            # 把 "['CCO']" / "['C','C','O']" 解析成真列表；失败就原样返回
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

        def _normalize_one_smi(smi):
            """把任意形态的 smi 归一成纯字符串 SMILES。"""
            if smi is None:
                return None

            if hasattr(smi, "tolist"):
                try:
                    smi = smi.tolist()
                except Exception:
                    smi = str(smi)

            if isinstance(smi, (list, tuple)):
                if len(smi) > 0 and all(isinstance(x, str) and len(x) == 1 for x in smi):
                    return ''.join(smi)
                if len(smi) == 1:
                    return _normalize_one_smi(smi[0])
                return ''.join(str(x) for x in smi)

            if isinstance(smi, str):
                smi = _maybe_strip_bytes_literal(smi)
                parsed = _maybe_parse_list_string(smi)
                if isinstance(parsed, (list, tuple)):
                    return _normalize_one_smi(parsed)
                return smi

            return str(smi)

        def _lookup_mol_graph(mol_graphs, smi_str):
            """尽量把 smi_str 映射到 mol_graphs 的某个 key，返回 (mol_graph, 使用的key)。"""
            cands = []
            s = _normalize_one_smi(smi_str)
            if s is None:
                raise KeyError("Empty SMILES")

            cands.append(s)
            cands.append(s.strip())

            if isinstance(s, str) and len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                cands.append(s[1:-1])

            parsed = _maybe_parse_list_string(s)
            if isinstance(parsed, (list, tuple)):
                cands.append(_normalize_one_smi(parsed))

            seen = set()
            uniq = []
            for k in cands:
                if k not in seen and k is not None:
                    seen.add(k)
                    uniq.append(k)

            for k in uniq:
                if k in mol_graphs:
                    return mol_graphs[k], k

            raise KeyError(f"SMILES key not found in mol_graphs. Tried candidates: {uniq!r}")

        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim
        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []

        # zero padding rows
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]

        fp_x_out = torch.empty((0, self.fp_fdim))

        # 注意：仍保留你的原始绝对路径；如需通用化可改为 os.path.join('dataset', data_name, 'raw', 'process_all.pkl')
        data_file = open(f'./dataset/processed/tg{tg_num}/TG{tg_num}_{split_tag}.pkl', 'rb')
        
        # data_file = open(f'./dataset/{data_name}/raw/process_all.pkl', 'rb')
        mol_graphs = pickle.load(data_file)
        data_file.close()

        mol_atom_num = []
        for smi in smiles:
            # ✅ 核心修复：不再用 smi[0]；按完整 SMILES 查字典，并做多候选回退
            mol_graph, used_key = _lookup_mol_graph(mol_graphs, smi)

            mol_atom_num.append(int(mol_graph.num_part[0][0]))
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

            fp_x_out = torch.cat((fp_x_out, mol_graph.fp_x))

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        # 安全截断，与你原始逻辑一致
        f_atoms = [row[:atom_fdim] if len(row) > atom_fdim else row for row in f_atoms]
        self.f_atoms = torch.FloatTensor(f_atoms)

        f_bonds = [row[:bond_fdim] if len(row) > bond_fdim else row for row in f_bonds]
        self.f_bonds = torch.FloatTensor(f_bonds)

        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None
        self.smiles = smiles
        self.mol_atom_num = mol_atom_num
        self.fp_x = fp_x_out

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        if atom_messages:
            f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=atom_messages)]
        else:
            f_bonds = self.f_bonds
        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        if self.b2b is None:
            b2b = self.a2b[self.b2a]
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()
            self.b2b = b2b * revmask
        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]
        return self.a2a

