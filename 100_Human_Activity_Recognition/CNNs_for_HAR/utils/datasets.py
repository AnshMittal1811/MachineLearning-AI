import os

import torch
from torch.utils.data import DataLoader, Dataset
from utils.helpers import load_file, load_group
from sklearn.preprocessing import StandardScaler

DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"har": "HumanActivityRecognition",
                 "newdataset": "NewDataset"}

DATASETS = list(DATASETS_DICT.keys())

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))

def get_data_size(dataset):
    """Return the correct data size."""
    return get_dataset(dataset).data_size

def get_num_classes(dataset):
    "Return the number of classes"
    return get_dataset(dataset).n_classes

def get_class_labels(dataset):
    """Return the class labels"""
    return get_dataset(dataset).classes

def get_dataloaders(dataset, root=None, shuffle=True, is_train=True, pin_memory=True,
                    batch_size=128, is_standardized=True, **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset : {"har"}
        Name of the dataset to load
    root : str
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    if root is None:
        dataset = Dataset(is_train=is_train,
                          is_standardized=is_standardized)
    else:
        dataset = Dataset(root=root,
                          is_train=is_train,
                          is_standardized=is_standardized)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)

class HumanActivityRecognition(Dataset):
    """Human activity recognition dataset"""
    data_size = (9, 128)
    n_classes = 6
    classes = ['walking',
               'walking upstairs',
               'walking downstairs',
               'sitting',
               'standing',
               'laying']

    def __init__(self, root=os.path.join(DIR, '../data/UCIHAR/'),
                 is_train=True,
                 is_standardized=False):
        """
        Parameters
        ----------

        root : string
            Path to the csv file with annotations.
        is_train : bool
            Chooses train or test set
        is_standardized : bool
            Chooses whether data is standardized
        """
        if is_train:
            image_set = 'train'
        else:
            image_set = 'test'

        data_train = self.load_dataset(root, 'train')
        if is_standardized and image_set == 'train':
            print("Loading Human Activity Recognition train dataset ...")
            X = self.standardize_data(data_train[0])
            self.X = torch.from_numpy(X).permute(0,2,1).float()
            self.Y = torch.from_numpy(data_train[1]).flatten().long()
        elif is_standardized and image_set == 'test':
            print("Loading Human Activity Recognition test dataset ...")
            data_test = self.load_dataset(root, 'test')
            X =  self.standardize_data(data_train[0], data_test[0])
            self.X = torch.from_numpy(X).permute(0, 2, 1).float()
            self.Y = torch.from_numpy(data_test[1]).flatten().long()
        else:
            print("Loading Human Activity Recognition %s dataset ..." % image_set)
            data = self.load_dataset(root, image_set)
            self.X = torch.from_numpy(data[0]).permute(0, 2, 1).float()
            self.Y = torch.from_numpy(data[1]).flatten().long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.X[idx,:,:]
        target = self.Y[idx]

        return input, target

    # load a dataset group, such as train or test
    # borrowed methods from the tutorial
    def load_dataset_group(self, group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
        # body acceleration
        filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
        # body gyroscope
        filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
        # load input data
        X = load_group(filenames, filepath)
        # load class output
        Y = load_file(prefix + group + '/y_' + group + '.txt')
        return X, Y

    # load the dataset, returns train and test X and y elements
    def load_dataset(self, root='', image_set='train'):
        # load all train
        X, Y = self.load_dataset_group(image_set, root)
        # zero-offset class values
        Y = Y - 1
        return X, Y

    # standardize data
    def standardize_data(self, X_train, X_test=None):
        """
        Standardizes the dataset

        If X_train is only passed, returns standardized X_train

        If X_train and X_test are passed, returns standardized X_test
        -------
        """
        # raise Exception("need to standardize the test set with the mean and stddev of the train set!!!!!!!")
        # remove overlap
        cut = int(X_train.shape[1] / 2)
        longX_train = X_train[:, -cut:, :]
        # flatten windows
        longX_train = longX_train.reshape((longX_train.shape[0] * longX_train.shape[1], longX_train.shape[2]))
        # flatten train and test
        flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))

        # standardize
        s = StandardScaler()
        # fit on training data
        s.fit(longX_train)
        # apply to training and test data
        if X_test is not None:
            print("Standardizing test set")
            flatX_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
            flatX_test = s.transform(flatX_test)
            flatX_test = flatX_test.reshape((X_test.shape))
            return flatX_test
        else:
            print("Standardizing train set")
            # reshape
            flatX_train = s.transform(flatX_train)
            flatX_train = flatX_train.reshape((X_train.shape))
            return flatX_train