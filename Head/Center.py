import numpy as np

from Head.Params import Params


class Center:
    def __init__(self, params: Params):
        self.params = params

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([])

    def load_model(self, dir_: str) -> None:
        print("Загрузка из файлов")

    def train_model(self):
        print("Обучение моделей")

    def test(self):
        print("Тестирование")
