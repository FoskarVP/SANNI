from Model.Networks import Base
from API import Snippet, Preprocessing
import numpy as np


class Classifier(Base):
    def __init__(self, size_subsequent: int, dataset: str, fraction: float, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.snippet_list = Preprocessing.search_snippet(data=dataset,
                                                         fraction=fraction,
                                                         size_subsequent=size_subsequent)
        print("Инициализации сверточной сети")

    def __load_model(self, dir_: str):
        print("Загрузка сверточной сети из файла")

    def create_predictor_dataset(self) -> str:
        return "Путь к датасету для прогноза"

    def predictor(self, data: np.ndarray) -> np.ndarray:
        return np.array([])

    def save_model(self, dir_: str) -> None:
        print("Загрузка")

    def get_snippet(self, class_snip: int) -> np.ndarray:
        return np.array([])

