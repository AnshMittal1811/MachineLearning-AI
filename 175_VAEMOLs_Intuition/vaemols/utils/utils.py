import numpy as np
import pandas as pd
from rdkit import Chem
import logging

def smiles_to_labels(smiles_data, char_to_int, max_length):
    labeled_data = np.zeros((len(smiles_data), max_length, 1), dtype=np.int32)
    smiles_data = [d.ljust(max_length) for d in smiles_data]
    for i in range(len(smiles_data)):
        for t, char in enumerate(smiles_data[i]):
            labeled_data[i, t, 0] = char_to_int[char]
    return labeled_data

def labels_to_smiles(labeled_data, int_to_char):
    return np.array([''.join([int_to_char[label] for label in labels]).strip(' ')
                     for labels in labeled_data], dtype=np.str)

def filter_smiles_to_mols(smiles_data):
    smiles_data_unique = pd.unique(smiles_data)
    valid_mols = []
    for smiles in smiles_data_unique:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_mols.append(mol)
        except:
            continue
    logging.info('Number of input smiles: ' + str(len(smiles_data)))
    logging.info('Number of unique input smiles: ' + str(len(smiles_data_unique)))
    logging.info('Number of unique valid mols: ' + str(len(valid_mols)))
    return valid_mols
