import numpy as np
import json
import os
from sklearn import metrics
from progress.bar import IncrementalBar
import time, sys
import matplotlib.pyplot as plt

from Model.Networks.Classifier import Classifier
from Model.Networks.Predictor import Predictor
from Model.Networks.Clear import Clear
from Head.Params import Params
from ResultBot.Bot import ResultBot
from API.Preprocessing import create_dataset
from API.Image import subsequent_to_image


class Center:
    def __init__(self, params: Params):

        self.bot = ResultBot(project_name="SANNI")
        self.bot.params["size_subsequent"] = params.size_subsequent
        self.bot.params["snippet_count"] = params.snippet_count
        self.bot.params["dataset_name"] = params.dataset_name

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.params = params
        create = False
        if os.path.exists(params.dir_dataset + "/current_params.json"):

            with open(params.dir_dataset + "/current_params.json") as f:
                current = json.load(f)
            if current["size_subsequent"] == self.params.size_subsequent \
                    and current["snippet_count"] == self.params.snippet_count:
                create = True
                print("Открываю созданый датасет")

        if not create:
            init_time = time.time()
            create_dataset(size_subsequent=params.size_subsequent,
                           dataset=params.dir_dataset,
                           snippet_count=params.snippet_count)
            mess = "Время создание датасета %s" % (time.time() - init_time)
            self.message(mess)

        self.classifier = Classifier(size_subsequent=params.size_subsequent,
                                     dataset=params.dir_dataset)
        self.predictor = Predictor(size_subsequent=params.size_subsequent,
                                   dataset=params.dir_dataset)
        self.clear = Clear(size_subsequent=params.size_subsequent,
                           dataset=params.dir_dataset)

        if not os.path.exists(self.params.dir_dataset + "/result"):
            os.mkdir(self.params.dir_dataset + "/result")

    def message(self, mess):
        self.bot.send_message(mess)
        print(mess)

    def predict(self, data: np.ndarray):  # -> np.ndarray:
        classifier = []
        y_classifier = self.classifier.predict(data)
        arr = []
        for i, item in enumerate(y_classifier):
            y_classifier = self.classifier.get_snippet(item)
            X_predict = np.stack([np.append(data[i], [0]), y_classifier])
            X_predict = X_predict.reshape(self.params.size_subsequent, 2)
            arr.append(X_predict)

        predict = self.predictor.predict(np.array(arr))
        """
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
        """
        return np.array(predict), np.array(classifier)

    def load_model(self, dir_: str) -> None:
        print("Загрузка из файлов")

    def train_model(self):
        mess = "Обучение моделей"
        self.message(mess)
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

            mess = "Открываю созданые датасеты"
            self.message(mess)

        if not model[0]:
            self.classifier.init_dataset()
            self.classifier.train_model()
            self.classifier.del_dataset()
        else:
            self.classifier.load_model()
        if not model[1]:
            self.predictor.init_dataset()
            self.predictor.train_model()
            self.predictor.del_dataset()

        else:
            self.predictor.load_model()

        if not model[2]:
            self.clear.init_dataset()
            self.clear.train_model()
            self.clear.del_dataset()

        else:

            self.clear.load_model()

    def general_test(self):
        mess = "Запуск генерального тестирования"
        self.message(mess)

        self.clear.init_dataset()
        y_predict_clear = self.clear.predict(self.clear.dataset.X_test)
        mess = "mse предсказателя без сниппета - {0}\n".format(
            metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                       y_pred=y_predict_clear))

        mess += "rmse предсказателя без сниппета- {0}".format(
            metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                       y_pred=y_predict_clear) * 0.5)

        y_predict, y_classifier = self.predict(self.clear.dataset.X_test)
        self.message(mess)

        mess = "mse предсказателя со сниппетом - {0};".format(
            metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                       y_pred=y_predict))
        mess += "rmse предсказателя со сниппетом- {0};".format(
            metrics.mean_squared_error(y_true=self.clear.dataset.y_test,
                                       y_pred=y_predict) * 0.5)

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

        plt.plot(self.clear.dataset.y_test,
                 label="true point",
                 linestyle=":",
                 marker="x")
        plt.plot(y_predict,
                 label="snippet point",
                 linestyle="-",
                 marker="o")
        plt.plot(y_predict_clear,
                 label="frame point",
                 linestyle="-.",
                 marker="+")
        plt.legend()
        plt.savefig(self.params.dir_dataset + '/result/general_test.png')
        plt.show()
        plt.clf()
        with open(self.params.dir_dataset + "/result/general_result.txt", 'w') as outfile:
            json.dump(result, outfile)

        self.message("Провел генеральное тестирование")


def test(self):
    self.message("Тестирование")
    self.classifier.init_dataset()
    self.classifier.test()
    self.classifier.del_dataset()
    self.predictor.init_dataset()
    self.predictor.test()
    self.clear.init_dataset()
    self.clear.test()
    self.general_test()
