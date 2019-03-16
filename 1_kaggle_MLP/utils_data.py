import os

from torch.utils.data import Dataset
import numpy as np
from typing import Sequence
from sklearn.model_selection import StratifiedShuffleSplit
import torch


def load_data(data_dir: str,
              which: str):
    """
    Loads data from a csv file
    :param data_dir: str
        Data directory path
    :param which: str
        Which data to load, train or test
    """
    assert which in ['train', 'test']

    if which == 'train':
        data = np.loadtxt(fname=os.path.join(data_dir, 'train_data.csv'), delimiter=',', skiprows=1)
        labels = np.loadtxt(fname=os.path.join(data_dir, 'train_labels.csv'), delimiter=',', skiprows=1)
        return data, labels
    elif which == 'test':
        data = np.loadtxt(fname=os.path.join(data_dir, 'test_data.csv'), delimiter=',', skiprows=1)
        return data


def stratified_split(data_dir: str):
    X = np.loadtxt(fname=os.path.join(data_dir, 'train_data.csv'), delimiter=',', skiprows=1)
    y = np.loadtxt(fname=os.path.join(data_dir, 'train_labels.csv'), delimiter=',', skiprows=1)
    print('Data loaded')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_index, test_index = next(sss.split(X, y))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dimensions = ','.join(list(map(str, list(range(X_train.shape[1])))))

    np.savetxt(os.path.join(data_dir, "train_splitted_data.csv"),
               X_train,
               delimiter=",",
               header=dimensions,
               comments='',
               fmt='%1.4f'
               )
    np.savetxt(os.path.join(data_dir, "train_splitted_labels.csv"),
               y_train,
               delimiter=",",
               header='label',
               comments='',
               fmt='%d'
               )
    np.savetxt(os.path.join(data_dir, "test_splitted_data.csv"),
               X_test,
               delimiter=",",
               header=dimensions,
               comments='',
               fmt='%1.4f'
               )
    np.savetxt(os.path.join(data_dir, "test_splitted_labels.csv"),
               y_test,
               delimiter=",",
               header='label',
               comments='',
               fmt='%d'
               )
    print('Data saved')


def save_prediction(data_dir: str,
                    prediction: Sequence[int],
                    file: str = 'submission.csv'):
    """
    Saves a sequence of predictions into a csv file with additional index column
    :param data_dir: str
        Data directory path
    :param prediction: Sequence of ints
        Predictions to save
    :param file: str
        Path to a file to save into
    """
    save_path = os.path.join(data_dir, file)
    pred_with_id = np.stack([np.arange(len(prediction)), prediction], axis=1)
    np.savetxt(fname=save_path, X=pred_with_id, fmt='%d', delimiter=',', header='id,label', comments='')


class Project1Dataset(Dataset):
    # __xs = []
    # __ys = []

    def __init__(self, data_dir: str, which: str):
        assert which in ['train', 'test']
        self.__xs = np.loadtxt(fname=os.path.join(data_dir, F'{which}_splitted_data.csv'), delimiter=',', skiprows=1)
        self.__ys = np.loadtxt(fname=os.path.join(data_dir, F'{which}_splitted_labels.csv'), delimiter=',', skiprows=1)

    def __getitem__(self, index):
        x = torch.from_numpy(self.__xs[index])
        y = torch.from_numpy(self.__ys[index])
        return x, y

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

