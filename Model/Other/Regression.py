from Model.Other.Func import Func
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class K_Regress(Func):

    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.model = KNeighborsRegressor(n_neighbors=10)
        self.name = "K_Regress"

    def train_model(self, send_message=print):
        self.model.fit(self.dataset.X_train, self.dataset.y_train)

