import numpy as np
from scipy.stats import mode
from Model.Other.Func import Func
from sklearn import preprocessing


class Mean(Func):

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.model = 0
        self.name = "Mean"

    def init_dataset(self):
        super().init_dataset()
        arr = np.loadtxt(self.dir_dataset + "/data_origin.txt")
        min_max_scaler = preprocessing.MinMaxScaler()
        data_norm = min_max_scaler.fit_transform(arr.reshape(-1, 1)).T[0]
        self.dataset.X_train = data_norm

    def train_model(self, send_message=print):
        self.model = np.mean(self.dataset.X_train)

    def predict(self, data: np.ndarray) -> np.ndarray:
        arr = np.full(len(data), self.model)
        return arr


class Median(Func):

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.model = 0
        self.name = "Median"

    def init_dataset(self):
        super().init_dataset()
        arr = np.loadtxt(self.dir_dataset + "/data_origin.txt")
        min_max_scaler = preprocessing.MinMaxScaler()
        data_norm = min_max_scaler.fit_transform(arr.reshape(-1, 1)).T[0]
        self.dataset.X_train = data_norm

    def train_model(self, send_message=print):
        self.model = np.median(self.dataset.X_train)

    def predict(self, data: np.ndarray) -> np.ndarray:
        arr = np.full(len(data), self.model)
        return arr


class Mode(Func):

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.model = 0
        self.name = "Mode"

    def init_dataset(self):
        super().init_dataset()
        arr = np.loadtxt(self.dir_dataset + "/data_origin.txt")
        min_max_scaler = preprocessing.MinMaxScaler()
        data_norm = min_max_scaler.fit_transform(arr.reshape(-1, 1)).T[0]
        self.dataset.X_train = data_norm

    def train_model(self, send_message=print):
        print(mode(self.dataset.X_train))
        self.model = mode(self.dataset.X_train)[0][0]

    def predict(self, data: np.ndarray) -> np.ndarray:
        arr = np.full(len(data), self.model)
        return arr
