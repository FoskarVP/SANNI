import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
import json

from Head.Params import Params


class DataSet:

    def __init__(self, dir_=None, bath_size=None, name=None) -> None:
        """
        Инициализация датасета
        :param dir_: Директория датасета
        :param bath_size: Размер пакета
        :param bath_size: Процент тестовой выборки
        """
        if dir_ is not None:
            self.bath_size = bath_size
            self.dir_ = dir_
            self.params = Params(dir_)
            read = zipfile.ZipFile(dir_ + "/dataset.zip", 'r')
            df = pd.read_csv(read.open(name + ".csv"), converters={"X": json.loads,
                                                                   "y": json.loads})

            X = np.stack(df.X.values)
            y = np.stack(df.y.values)

            if self.params.shuffle:
                index = np.array(range(len(X)))
                self.i_train, self.i_test = train_test_split(index,
                                                             test_size=self.params.percent_test,
                                                             random_state=self.params.random)
                self.i_train, self.i_valid, = train_test_split(self.i_train,
                                                               test_size=self.params.percent_test,
                                                               random_state=self.params.random)

                self.X_train = X[self.i_train]
                self.X_valid = X[self.i_valid]
                self.X_test = X[self.i_test]
                self.y_train = y[self.i_train]
                self.y_valid = y[self.i_valid]
                self.y_test = y[self.i_test]
            else:
                self.X_train = X[:int(X.shape[0] * 0.6)]
                self.X_valid = X[int(X.shape[0] * 0.6):int(X.shape[0] * 0.75)]
                self.X_test = X[int(X.shape[0] * 0.75):]
                self.y_train = y[:int(X.shape[0] * 0.6)]
                self.y_valid = y[int(X.shape[0] * 0.6):int(X.shape[0] * 0.75)]
                self.y_test = y[int(X.shape[0] * 0.75):]
            self.n = len(self.X_train)
            self.cur_index = 0
            self.count_ep = 0
            print("Загрузил датасет")

    def next_batch(self, random=False, type_batch="train") -> np.ndarray:
        """
        Получить следующую порцию датасета
        :param random: случайность следующего пакета
        :param type_batch: тип пакате (test - тестовая, valid - валидационная, train - обучающая)
        :return: Массив сэмплов
        """
        print("Возражаю следующий пакет")
        yield np.array([])

    def next_sample(self, random=False, type_batch="train") -> np.ndarray:
        """
        Получить следующую cэмпл
        :param random: случайность следующего пакета
        :param type_batch: тип пакате (test - тестовая, valid - валидационная, train - обучающая)
        :return: Сэмпл ввиде массива
        """
        print("Возражаю следующий пакет")
        yield np.array([])
