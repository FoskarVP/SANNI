import numpy as np


class DataSet:

    def __init__(self, dir_: str, bath_size: int) -> None:
        """
        Инициализация датасета
        :param dir_: Директория датасета
        :param bath_size: Размер пакета
        """
        self.bath_size = bath_size
        self.dir_ = dir
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
