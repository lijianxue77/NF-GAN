import torch.utils.data
import pandas
import torch


class Traffic_Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path, dataset_features, dataset_dtypes, generated_features, batch_size=32, transform=None):
        self.transform = transform
        self.file_path = file_path
        self.batch_size = batch_size
        self.columns = dataset_features
        self.dtypes = dataset_dtypes
        self.dataframe = pandas.read_csv(self.file_path, names=self.columns, header=None)

        self.categorical_features = generated_features
        self.categorical_values = {}
        self.feature_bits_dict = {}
        self.vocabulary_length = 0
        self.max_seq_length = 0

        self.init_values()

    def __len__(self):
        return len(self.dataframe.index)-self.batch_size

    def __getitem__(self, index):
        list_of_dict = torch.zeros(self.batch_size, self.max_seq_length).type(torch.LongTensor)
        for i, idx in enumerate(range(index, index+self.batch_size)):
            dict = self.dataframe.iloc[idx, :].to_dict()
            if self.transform is not None:
                list_of_dict[i, :] = self.transform(self, dict)
        return list_of_dict

    def init_values(self):
        for categorical_feature in self.categorical_features:
            self.categorical_values[categorical_feature] = self.dataframe[categorical_feature].unique()

        for feature in self.categorical_features:
            self.feature_bits_dict[feature] = len(self.categorical_values[feature])

        for feature in self.feature_bits_dict.keys():
            self.vocabulary_length += self.feature_bits_dict[feature]
        self.max_seq_length = len(self.categorical_features)
