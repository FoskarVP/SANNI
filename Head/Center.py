import numpy as np
import json
import os
from sklearn import metrics
from progress.bar import IncrementalBar
import time, sys

from Model.Networks.Classifier import Classifier
from Model.Networks.Predictor import Predictor
from Model.Networks.Clear import Clear
from Head.Params import Params

from API.Preprocessing import create_dataset
from API.Image import subsequent_to_image


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
            init_time = time.time()
            create_dataset(size_subsequent=params.size_subsequent,
                           dataset=params.dir_dataset,
                           fraction=params.fraction)
            print("Время инициализации датасета %s" % (time.time() - init_time))

        self.classifier = Classifier(size_subsequent=params.size_subsequent,
                                     dataset=params.dir_dataset)
        self.predictor = Predictor(size_subsequent=params.size_subsequent,
                                   dataset=params.dir_dataset)
        self.clear = Clear(size_subsequent=params.size_subsequent,
                           dataset=params.dir_dataset)

        if not os.path.exists(self.params.dir_dataset + "/result"):
            os.mkdir(self.params.dir_dataset + "/result")

    def predict(self, data: np.ndarray):  # -> np.ndarray:
        predict = []
        classifier = []
        bar = IncrementalBar('General test', max=len(data))

        for j, i in enumerate(data):
            bar.next()
            y_classifier = self.classifier.predict(np.array([i]))[0]
            classifier.append(y_classifier)
            y_classifier = self.classifier.get_snippet(y_classifier)
            X_predict = np.stack([np.append(i, [0]), y_classifier])
            X_predict = X_predict.reshape(self.params.size_subsequent, 2)
            y_predict = self.predictor.predict(np.array([X_predict]))
            _predict = self.clear.predict(np.array([i]))
            predict.append(y_predict[0][0])

        bar.finish()
        return np.array(predict), np.array(classifier)

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

    def general_test(self):
        print("Запуск генерального тестирования")

        y_predict_clear = self.clear.predict(self.clear.dataset.X_test)
        print("mse предсказателя без сниппета - {0};".
              format(metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                                y_pred=y_predict_clear)))
        print("rmse предсказателя без сниппета- {0};".
              format(metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                                y_pred=y_predict_clear) * 0.5))

        y_predict, y_classifier = self.predict(self.clear.dataset.X_test)

        print("mse предсказателя со сниппетом - {0};".
              format(metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                                y_pred=y_predict)))
        print("rmse предсказателя со сниппетом- {0};".
              format(metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                                y_pred=y_predict) * 0.5))

        result = {
            "mse_sn": metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                                 y_pred=y_predict),
            "rmse_sn": metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                                  y_pred=y_predict) * 0.5,
            "mse": metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                              y_pred=y_predict_clear),
            "rmse": metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                               y_pred=y_predict_clear) * 0.5
        }
        with open(self.params.dir_dataset + "/result/general_result.txt", 'w') as outfile:
            json.dump(result, outfile)

        print("Провел генеральное тестирование")

    def test(self):
        print("Тестирование")
        self.classifier.test()
        self.predictor.test()
        self.clear.test()
        self.general_test()
