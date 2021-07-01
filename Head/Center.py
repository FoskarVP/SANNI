import numpy as np
import json
import os

from Model.Networks.Classifier import Classifier
from Model.Networks.Predictor import Predictor
from Model.Networks.Clear import Clear
from Head.Params import Params
from API.Preprocessing import create_dataset


class Center:
    def __init__(self, params: Params):
        self.params = params
        create = False
        if os.path.exists(params.dir_dataset + "/current_params.json"):

            with open(params.dir_dataset + "/current_params.json") as f:
                current = json.load(f)
            if current["size_subsequent"] == self.params.size_subsequent \
                    and current["fraction"] == self.params.fraction:
                create = True
                print("Открываю созданый датасет")

        if not create:
            create_dataset(size_subsequent=params.size_subsequent,
                           dataset=params.dir_dataset,
                           fraction=params.fraction)
        self.classifier = Classifier(size_subsequent=params.size_subsequent,
                                     dataset=params.dir_dataset)
        self.predictor = Predictor(size_subsequent=params.size_subsequent,
                                   dataset=params.dir_dataset)
        self.clear = Clear(size_subsequent=params.size_subsequent,
                                   dataset=params.dir_dataset)

        if not os.path.exists(self.params.dir_dataset + "/result"):
            os.mkdir(self.params.dir_dataset + "/result")

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([])

    def load_model(self, dir_: str) -> None:
        print("Загрузка из файлов")

    def train_model(self):
        print("Обучение моделей")
        model = [False, False, False]
        if os.path.exists(self.params.dir_dataset + "/current_params.json"):

            with open(self.params.dir_dataset + "/current_params.json") as f:
                current = json.load(f)
            if current["classifier"]:
                model[0] = True
            if current["predict"]:
                model[1] = True
            if current["clear"]:
                model[2] = True
            print("Открываю созданый датасет")

        if not model[0]:
            self.classifier.train_model()
        else:
            self.classifier.load_model()

        if not model[1]:
            self.predictor.train_model()
        else:
            self.predictor.load_model()

        if not model[2]:
            self.clear.train_model()
        else:
            self.clear.load_model()

    def test(self):
        print("Тестирование")
        self.classifier.test()
        self.predictor.test()
        self.clear.test()