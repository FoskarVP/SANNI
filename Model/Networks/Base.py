from abc import ABC, abstractmethod
from Model import Dataset
import numpy as np


class BaseModel():

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        """
        Инициализациия модели
        :param size_subsequent: размер подпоследовательности - int
        :param dataset: путь к датасету - str
        :param load: Путь к файлу модели = str
        """
        self.dataset = Dataset.DataSet(dataset, 32)
        self.size_subsequent = size_subsequent
        pass

    def __load_model(self, dir_: str):
        """
        Загрузка модели
        :param dir_: директориЯ где хранится модель -str
        """
        pass

    def save_model(self, dir_: str) -> None:
        """
        Сохранения модели в файл
        :param dir_: Директория датасета - str
        """
        pass

    def predictor(self, data: np.ndarray) -> np.ndarray:
        """
        Сохранения модели в файл
        :param data: Входная последовательномть - np.ndarray
        """
        pass

    def train(self):
        print("Провел обучение")

    def test(self):
        print("Провел внутренние тестирование")
