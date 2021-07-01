from abc import ABC, abstractmethod
from Model.Dataset import DataSet
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt


class BaseModel:

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        """
        Инициализациия модели
        :param size_subsequent: размер подпоследовательности - int
        :param dataset: путь к датасету - str
        :param load: Путь к файлу модели = str
        """
        self.dir_dataset = dataset
        self.epochs = 0
        self.bath_size = 0
        self.pool = 0
        self.dataset = DataSet()
        self.size_subsequent = size_subsequent
        self.model = Sequential()
        pass

    def __load_model(self, dir_: str):
        """
        Загрузка модели
        :param dir_: директориЯ где хранится модель -str
        """
        pass

    def save_model(self) -> None:
        """
        Сохранения модели в файл
        """
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Сохранения модели в файл
        :param data: Входная последовательномть - np.ndarray
        :return: массив ответов
        """
        return self.model.predict(data)

    def test(self):
        print("Провел внутренние тестирование")

