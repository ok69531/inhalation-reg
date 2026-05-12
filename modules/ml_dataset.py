import os
import pickle
import logging
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint, Descriptors

logger = logging.getLogger(__name__)


def load_dataset(*, root = '../dataset', tg_num = 403, fp_type = 'maccs', log_transform = True):
    file_name = f'ml_tg{tg_num}.pkl'
    data_path = os.path.join(root, 'processed', file_name)
    
    if os.path.exists(data_path):
        logger.info('File Exsit')
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        logger.info('File not Exist')
        logger.info('Preprocessing data...')
        dataset = process_dataset(root = os.path.join(root, 'raw'), tg_num = tg_num)
    
    data_list = dataset['data']
    
    smiles = []
    y = []
    fingerprints = []
    descriptors = []
    for i in range(len(data_list)):
        smiles.append(data_list[i]['smiles'])
        y.append(data_list[i]['y'])
        fingerprints.append(data_list[i][fp_type])
        descriptors.append(data_list[i]['descriptor'])
    
    fingerprints = np.stack(fingerprints)
    descriptors = pd.DataFrame(descriptors)
    valid_cols = descriptors.columns[~descriptors.isna().any(axis=0)]
    descriptors = descriptors[valid_cols]
    descriptors = descriptors.to_numpy()
    
    x = np.concatenate([fingerprints, descriptors], axis = 1)
    y = np.array(y)
    
    if log_transform:
        y = np.log10(y)
    
    return x, y, smiles
    
    
def process_dataset(*, root = '../dataset/raw', tg_num = 403):
    file_map = {
        403: "tg403.xlsx",
        412: "tg412_413.xlsx",
    }

    if tg_num not in file_map:
        raise ValueError(f"tg_num must be one of {list(file_map.keys())}, got {tg_num}")

    file_name = file_map[tg_num]
    data_path = os.path.join(root, file_name)
    
    df = pd.read_excel(data_path)
    
    raw_y = df.value.to_numpy()
    raw_smiles = df.smiles
    raw_mols = [Chem.MolFromSmiles(x) for x in raw_smiles]
    
    mol_idx = [bool(x) for x in raw_mols]
    
    y = raw_y[mol_idx]
    smiles = raw_smiles[mol_idx]
    mols = list(filter(None, raw_mols))
    
    data = []
    for smi, yi, mol in zip(smiles, y, mols):
        item = {
            'smiles': smi, 
            'y': yi
        }
        item.update(smiles2fing(mol))
        data.append(item)
    
    if len(data) == 0:
        raise ValueError("No valid molecules found.")
    
    processed_data = {
        'smiles_bool_idx': mol_idx,
        'data': data
    }
    
    save_path = f'../dataset/processed'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f'ml_tg{tg_num}.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data


def smiles2fing(mol):
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=int)
    morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=int)
    rdkit = np.array(RDKFingerprint(mol), dtype=int)
    layered = np.array(AllChem.LayeredFingerprint(mol), dtype=int)
    pattern = np.array(AllChem.PatternFingerprint(mol), dtype=int)
    descriptor = Descriptors.CalcMolDescriptors(mol)
    
    fingerprints = {
        'maccs': maccs,
        'morgan': morgan,
        'rdkit': rdkit,
        'layered': layered,
        'pattern': pattern,
        'descriptor': descriptor
    }
    
    return fingerprints
