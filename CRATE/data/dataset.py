import sklearn
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(data, size, transform_train, transform_test, data_dir=None):
    if data_dir is None:
        data_dir = "../" + data
    if data == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "flower":
        trainset = torchvision.datasets.Flowers102(root=data_dir, split="train", download=True,
                                                   transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.Flowers102(root=data_dir, split="test", download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "pets":
        trainset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True,
                                                      transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="test", download=True,
                                                     transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    return trainset, testset


class T1SNPDataset(Dataset):
    def __init__(self, t1_data_path, cognitive_data_path, snp_data_path, target='PHC_MEM'):
        self.target = target

        self.t1_data = pd.read_csv(t1_data_path)
        self.cognitive_data = pd.read_csv(cognitive_data_path)
        self.snp_data = pd.read_csv(snp_data_path)

        self.T1_FEATURES = self.t1_data.columns.values[9:-1]
        self.SNP_FEATURES = self.snp_data.columns.values[7:-1]
        self.t1_data_num_features = len(self.T1_FEATURES)
        self.snp_data_num_features = len(self.SNP_FEATURES)

        self.all_data = pd.merge(self.t1_data, self.snp_data, on="RID")
        self.all_features = np.concatenate((self.T1_FEATURES, self.SNP_FEATURES))
        self.snp_features = self.SNP_FEATURES
        self.t1_features = self.T1_FEATURES

        tensor_dataset = pd.merge(self.all_data, self.cognitive_data, on='RID')
        self.dataset = tensor_dataset[np.concatenate((self.all_features, [self.target]))].dropna()

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        t1_features = item[self.T1_FEATURES].values
        snp_features = item[self.SNP_FEATURES].values
        return t1_features, snp_features, item[self.target]

    @staticmethod
    def prepare_train_val_sets(datamodule, task='regression', train_size=0.8, random_seed=43):
        X_all = datamodule.dataset[datamodule.all_features].astype('float32').values
        if task == 'regression':
            y_all = datamodule.dataset[datamodule.target].astype('float32').values
        else:
            encoder = LabelEncoder()
            y_all = encoder.fit_transform(datamodule.dataset[datamodule.target].values)
        X = dict()
        y = dict()

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, train_size=train_size, random_state=random_seed
        )

        X_train_mr = X_train[:, :datamodule.t1_data_num_features]
        X_train_snp = X_train[:, datamodule.t1_data_num_features:]
        X_test_mr = X_test[:, :datamodule.t1_data_num_features]
        X_test_snp = X_test[:, datamodule.t1_data_num_features:]

        preprocess = sklearn.preprocessing.StandardScaler()
        preprocess.fit(X_train_mr)
        X_train_mr = preprocess.transform(X_train_mr)
        X_test_mr = preprocess.transform(X_test_mr)
        X_train = np.concatenate((X_train_mr, X_train_snp), axis=1)
        X_test = np.concatenate((X_test_mr, X_test_snp), axis=1)

        # create train and validation pytorch datasets from the numpy arrays
        datamodule.train = T1SNPSubset(datamodule.t1_features, datamodule.snp_features, torch.tensor(X_train), torch.tensor(y_train[:, np.newaxis]))
        datamodule.val = T1SNPSubset(datamodule.t1_features, datamodule.snp_features, torch.tensor(X_test), torch.tensor(y_test[:, np.newaxis]))

        return datamodule


class T1SNPSubset(Dataset):
    def __init__(self, T1_features, SNP_features, X, y):
        self.T1_FEATURES = T1_features
        self.SNP_FEATURES = SNP_features

        self.t1_data_num_features = len(self.T1_FEATURES)
        self.snp_data_num_features = len(self.SNP_FEATURES)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        t1_features = self.X[index, :self.t1_data_num_features].unsqueeze(1)
        snp_features = self.X[index, self.t1_data_num_features:].unsqueeze(1)
        return t1_features, snp_features, self.y[index]
