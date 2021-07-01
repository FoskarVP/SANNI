import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
import json


class DataSet:

    def __init__(self, dir_=None, bath_size=None, name=None, percent=None) -> None:
        """
        Инициализация датасета
        :param dir_: Директория датасета
        :param bath_size: Размер пакета
        :param bath_size: Процент тестовой выборки
        """
        if dir_ is not None:
            self.bath_size = bath_size
            self.dir_ = dir_
            read = zipfile.ZipFile(dir_ + "/dataset.zip", 'r')
            ## костыль

            df = pd.read_csv(read.open(name + ".csv"), converters={"X": json.loads,
                                                                   "y": json.loads})

            X = np.stack(df.X.values)
            y = np.stack(df.y.values)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                    y,
                                                                                    test_size=percent,
                                                                                    random_state=42)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train,
                                                                                      self.y_train,
                                                                                      test_size=0.33,
                                                                                      random_state=42)
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
