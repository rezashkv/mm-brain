import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class T1SNPDataset(Dataset):
    def __init__(self, t1_data_path, cognitive_data_path, snp_data_path, target='PHC_MEM'):
        self.target = target

        self.t1_data = pd.read_csv(t1_data_path)
        self.cognitive_data = pd.read_csv(cognitive_data_path)
        self.snp_data = pd.read_csv(snp_data_path)

        self.T1_FEATURES = self.t1_data.columns.values[9:-1]
        self.SNP_FEATURES = self.snp_data.columns.values[1:]
        self.t1_data_num_features = len(self.T1_FEATURES)
        self.snp_data_num_features = len(self.SNP_FEATURES)

        self.all_data = pd.merge(self.t1_data, self.snp_data, left_on='CBICA_ID', right_on='Patient_ID')
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

