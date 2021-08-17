from Model.Other.Func import Func
import rpy2.robjects as R
from rpy2.robjects.packages import importr
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt

import json

import numpy as np


class inputeTS(Func):

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.name = "inputTS"
        self.imputeTS = importr("imputeTS")
        self.na_interpolation = R.r["na_interpolation"]

    def predict(self, data: np.ndarray) -> np.ndarray:
        self.init_dataset()
        arr = np.loadtxt(self.dir_dataset + "/data_origin.txt")
        min_max_scaler = preprocessing.MinMaxScaler()
        data_norm = min_max_scaler.fit_transform(arr.reshape(-1, 1)).T[0]
        index = self.dataset.i_test+(self.size_subsequent-1)
        data_norm[index] = np.nan
        res4 = R.FloatVector(data_norm.tolist())
        X_model = self.na_interpolation(res4)
        vector = np.array(X_model)[index]
        return vector

    def test(self, send_message=print):
        arr = np.loadtxt(self.dir_dataset + "/data_origin.txt")
        min_max_scaler = preprocessing.MinMaxScaler()
        data_norm = min_max_scaler.fit_transform(arr.reshape(-1, 1)).T[0]
        index = np.array(range(len(data_norm)))
        X_train, X_test = train_test_split(index, test_size=0.25, random_state=42,shuffle=False)
        X = data_norm.copy()
        X[X_test] = np.nan
        res4 = R.FloatVector(X.tolist())
        X_model = self.na_interpolation(res4)
        vector = np.array(X_model)[X_test]
        data_norm = data_norm[X_test]

        result = {
            "mse": metrics.mean_squared_error(data_norm, y_pred=vector),
            "rmse": metrics.mean_squared_error(data_norm, y_pred=vector) * 0.5
        }
        with open(self.dir_dataset + "/result/{0}_result.txt".format(self.name), 'w') as outfile:
            json.dump(result, outfile)

        send_message("Провел внутренние тестирование {0}\n{1}".format(self.name, result))
        return vector
