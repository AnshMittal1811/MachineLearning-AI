import argparse
import urllib
import gzip
import shutil
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--file_url', type=str, default='ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_24_1_chemreps.txt.gz',
                    help='CheMBL24 chemreps file link')
parser.add_argument('--file_name', type=str, default='chembl_24.txt', help='Data file name')
parser.add_argument('--file_dir', type=str, default='data/', help='Data file directory')
parser.add_argument('--max_length', type=int, default=120, help='Max length')

def main():
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    file_path = args.file_dir + args.file_name
    try:
        logging.info('Reading file.')
        data = pd.read_table(file_path)
    except FileNotFoundError:
        logging.info('Retrieving file from link since file not found...')
        zip_file_path = file_path+'.gz'
        urllib.request.urlretrieve(args.file_url, zip_file_path)
        logging.info('File retrieved.')
        logging.info('Unzipping...')
        zip_file_path = file_path+'.gz'
        with gzip.open(zip_file_path, 'rb') as zf:
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(zf, f)
        logging.info('File saved as '+file_path)
        logging.info('Reading file.')
        data = pd.read_table(file_path)

    smiles_data = data['canonical_smiles']
    smiles_data = np.array(smiles_data).reshape(-1)
    logging.info('Number of mols: '+str(len(smiles_data)))
    idx = [i for i, x in enumerate(smiles_data) if len(x)<=120]
    logging.info('Number of valid mols: '+str(len(idx)))
    smiles_data = smiles_data[idx]

    logging.info('Getting a unique character set...')
    char_set = set()
    for i in tqdm(range(len(smiles_data))):
        smiles_data[i] = smiles_data[i].ljust(args.max_length)
        char_set = char_set.union(set(smiles_data[i]))
    char_set_list = sorted(list(char_set))
    logging.info('Number of characters: '+str(len(char_set_list)))

    char_to_int = dict((c, i) for i, c in enumerate(char_set))
    int_to_char = dict((i, c) for i, c in enumerate(char_set))

    logging.info('Converting char to int...')
    labeled_data = np.zeros((len(smiles_data), args.max_length, 1), dtype=np.int32)
    for i in tqdm(range(len(smiles_data))):
        for t, char in enumerate(smiles_data[i]):
            labeled_data[i, t, 0] = char_to_int[char]

    logging.info('Train test split to 0.8:0.2')
    x_train, x_test = train_test_split(labeled_data, test_size=0.2)
    logging.info('Number of training data: '+str(len(x_train)))
    logging.info('Number of testing data: '+str(len(x_test)))

    logging.info('Saving...')
    np.save(args.file_dir + 'x_train', x_train)
    np.save(args.file_dir + 'x_test', x_test)
    np.savez(args.file_dir + 'char_data', char_to_int=char_to_int, int_to_char=int_to_char)
    logging.info('Saved.')

if __name__ == "__main__":
    main()
