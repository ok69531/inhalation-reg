import re
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import rdkit.Chem as Chem

from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data, InMemoryDataset


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}


extended_periodic_table = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 
    110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}
extended_periodic_table = {key - 1: value for key, value in extended_periodic_table.items()}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))


class InhaleRegDataset(InMemoryDataset):
    def __init__(
        self, 
        root, assay_name = None, tg_num = None, 
        split = None, test_size = 0.2, random_state = 42, 
        removeHs = False, transform = None, pre_transform = None, pre_filter = None
    ):
        """
            Examples
            --------
            # 2) 특정 assay만 전체 로드
            ds_assay = InhaleRegDataset(
                root='data',
                assay_name='ACEA_T47D_80hr_Negative'
            )

            # 3) 특정 assay train/test 분리 로드
            ds_train = InhaleRegDataset(
                root='data',
                assay_name='ACEA_T47D_80hr_Negative',
                split='train',
                test_size=0.2,
                random_state=42,
            )
            
            # 4) tg
            ds_train = InhaleRegDataset(
                root='data',
                tg_num = 421,
                expose_type = 'mgkg',
                split='train',
                test_size=0.2,
                random_state=42,
            )
        """
        self.root = root
        self.tg = tg_num
        # self.expose_type = expose_type
        if self.tg not in [403, 412]:
            raise ValueError('Invalid TG Number. Only support 403 and 412')
        
        if self.tg is not None:
            self.assay_name = 'value'
        else:
            self.assay_name = assay_name
        
        self.split = split
        self.test_size = test_size
        self.random_state = random_state
        self.removeHs = removeHs
        
        if self.tg is None:
            self.data_root = 'TC_REG'
        elif self.tg:
            if self.tg == 403: self.data_root = f'tg{tg_num}'
            elif self.tg == 412: self.data_root = f'tg{tg_num}'
        else:
            raise ValueError('Unsupported Data Type! Use tier or tg_num.')
        
        if self.split not in {None, 'train', 'test'}:
            raise ValueError("split must be one of {None, 'train', 'test'}")
        
        if self.assay_name is None and self.split is not None:
            raise ValueError("split is only supported when assay_name is specified.")

        super(InhaleRegDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.data_root)
    
    @property
    def processed_file_names(self):
        assay_tag = 'all' if self.assay_name is None else self.assay_name
        split_tag = 'full' if self.split is None else self.split
        
        if self.tg is None:
            filename = f'ToxCast_{assay_tag}_{split_tag}.pt'
        else:
            filename = f'TG{self.tg}_{split_tag}.pt'
        return [filename]

    @property
    def num_classes(self):
        return 1
    
    def _load_raw_dataset(self):
        if self.tg is None:
            self.raw_data = pd.read_csv(os.path.join(self.raw_dir, 'data.csv'))
        else:
            if self.tg == 403: file_name = f'tg{self.tg}.xlsx'
            elif self.tg == 412: file_name = f'tg{self.tg}_413.xlsx'
            self.raw_data = pd.read_excel(os.path.join(self.raw_dir, file_name))
        self.raw_data = self.raw_data.dropna(subset=['smiles']).copy()
        self.raw_data = self.raw_data[self.raw_data['smiles'] != ''].reset_index(drop=True)
        if 'smiles' not in self.raw_data.columns:
            raise ValueError("Input File must contain a 'SMILES' column.")
    
    def _smiles_screen(self):
        mols = [Chem.MolFromSmiles(x) for x in self.raw_data.smiles]
        self.mols_na_idx = [i for i in range(len(mols)) if mols[i] is None]
        self.data = self.raw_data.drop(self.mols_na_idx).reset_index(drop = True)
        if self.mols_na_idx:
            print(f'Removed SMILES Index: {self.mols_na_idx}')
        
        self.mols = filter(None, mols)
        self.all_assays = list(self.data.columns.drop(['casrn', 'smiles']))
    
    def _select_assay_and_split(self):
        # 1) 전체 assay 한 번에 로드
        if self.assay_name is None:
            self.target_assays = self.all_assays
            self.data = self.data[['smiles'] + self.target_assays].reset_index(drop=True)

        # 2) 특정 assay 하나만 로드
        else:
            if self.assay_name not in self.all_assays:
                raise ValueError(
                    f"'{self.assay_name}' is not in assays.\n"
                    f"Available assays: {self.all_assays}"
                )

            self.target_assays = [self.assay_name]

            # 단일 assay는 해당 assay label 없는 행 제거
            self.data = (
                self.data[['smiles', self.assay_name]]
                .dropna(subset=[self.assay_name])
                .reset_index(drop=True)
            )

            # 3) assay별 train/test split
            if self.split is not None:
                y = self.data[self.assay_name]

                # stratify_y = None
                # if self.use_stratify:
                #     value_counts = y.value_counts()
                #     # stratify는 각 클래스가 최소 2개 이상 있어야 안전
                #     if y.nunique() > 1 and (value_counts >= 2).all():
                #         stratify_y = y

                train_df, test_df = train_test_split(
                    self.data,
                    test_size=self.test_size,
                    shuffle=True,
                    random_state=self.random_state,
                    # stratify=stratify_y,
                )

                if self.split == 'train':
                    self.data = train_df.reset_index(drop=True)
                else:
                    self.data = test_df.reset_index(drop=True)

        self.smiles = self.data['smiles'].tolist()
        self.target = self.data[self.target_assays].to_numpy()
    
    def smiles_to_graph(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError('Invalid SMILES string')
        mol = mol if self.removeHs else Chem.AddHs(mol)
        
        x = _atom_feature(mol)
        
        edge_index = []
        edge_feature_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))
            edge_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
            edge_feature_list.append(edge_feature)
            edge_feature_list.append(edge_feature)

        x = torch.tensor(x, dtype = torch.int32)
        edge_index = torch.tensor(edge_index, dtype = torch.long).t()
        edge_attr = torch.tensor(edge_feature_list, dtype = torch.long)
        
        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)
        
        return data

    def _construct_graphs(self):
        data_list = []
        for i in range(len(self.data)):
            smiles = self.smiles[i]
            data = self.smiles_to_graph(smiles)
            data.origin_y = torch.tensor(self.target[i]).view(-1)
            data.y = torch.tensor(np.log10(self.target[i])).view(-1)
            # data.y = torch.tensor(self.target[i]).to(torch.long).view(-1)
            data.smiles = smiles
            data.idx = i
            
            data_list.append(data)
        
        return data_list
    
    def process(self):
        self._load_raw_dataset()
        self._smiles_screen()
        self._select_assay_and_split()
        data_list = self._construct_graphs()
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(self.collate(data_list), self.processed_paths[0])


def _atom_feature(mol):
        atom_feature_list = []
        for atom in mol.GetAtoms():
            atom_feature = [
                safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
                safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
                allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
            atom_feature_list.append(atom_feature)

        atom_feature_list = np.array(atom_feature_list)
        
        return atom_feature_list


if __name__ == '__main__':
    root = '../dataset'
    dataset = InhaleRegDataset(root, tg_num=403)
    
    print(dataset[0])
    print(dataset[0].y)
    print(dataset[0].origin_y)

    train_dataset = InhaleRegDataset(root, tg_num=403, split = 'train')
    test_dataset = InhaleRegDataset(root, tg_num=403, split = 'test')
