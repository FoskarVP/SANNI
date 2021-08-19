import numpy as np
import json
import os
from sklearn import metrics
import time
import matplotlib.pyplot as plt

from Model.Networks.Classifier import Classifier
from Model.Networks.Predictor import Predictor, Predictor_label
from Model.Networks.Clear import Clear
from Model.Other.Mean import Mean, Median, Mode

from Head.Params import Params
from Model.Other.Regression import K_Regress
from Model.Other.imputeTS import inputeTS
from ResultBot.Bot import ResultBot
from API.Preprocessing import create_dataset

from Head.const import CURRENT_PARAMS_DIR


class Center:
    def __init__(self, params: Params, bot=None):
        if bot is None:
            self.bot = ResultBot(project_name="SANNI")
        else:
            self.bot = bot
        self.bot.params["size_subsequent"] = params.size_subsequent
        self.bot.params["snippet_count"] = params.snippet_count
        self.bot.params["dataset_name"] = params.dataset_name
        self.bot.send_message("Запуск обучения")

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.params = params
        create = False
        clear_create = False
        if os.path.exists(params.dir_dataset + CURRENT_PARAMS_DIR):

            with open(params.dir_dataset + CURRENT_PARAMS_DIR) as f:
                current = json.load(f)
            if current["size_subsequent"] == self.params.size_subsequent \
                    and current["snippet_count"] == self.params.snippet_count:
                if current["size_subsequent"] == self.params.size_subsequent:
                    clear_create = True
                create = True
                print("Открываю созданый датасет")

        if not create:
            init_time = time.time()
            create_dataset(size_subsequent=params.size_subsequent,
                           dataset=params.dir_dataset,
                           snippet_count=params.snippet_count)
            mess = "Время создание датасета %s" % (time.time() - init_time)
            if clear_create:
                with open(params.dir_dataset + CURRENT_PARAMS_DIR) as f:
                    current = json.load(f)
                    current["clear"] = True
                    with open(params.dir_dataset + CURRENT_PARAMS_DIR, 'w') as outfile:
                        json.dump(current, outfile)
            self.message(mess)

        self.models = {"classifier": Classifier(size_subsequent=params.size_subsequent,
                                                dataset=params.dir_dataset),
                       "predictor": Predictor(size_subsequent=params.size_subsequent,
                                              dataset=params.dir_dataset),
                       "clear": Clear(size_subsequent=params.size_subsequent,
                                      dataset=params.dir_dataset),
                       "Mean": Mean(size_subsequent=params.size_subsequent,
                                    dataset=params.dir_dataset),
                       "Median": Median(size_subsequent=params.size_subsequent,
                                        dataset=params.dir_dataset),
                       "Mode": Mode(size_subsequent=params.size_subsequent,
                                    dataset=params.dir_dataset),
                       "K_Regress": K_Regress(size_subsequent=params.size_subsequent,
                                              dataset=params.dir_dataset),
                       "inputeTS": inputeTS(size_subsequent=params.size_subsequent,
                                            dataset=params.dir_dataset)
                       }

        if not os.path.exists(self.params.dir_dataset + "/result"):
            os.mkdir(self.params.dir_dataset + "/result")

    def message(self, mess):
        mess = str(mess)
        self.bot.send_message(mess)
        print(mess)

    def predict(self, data: np.ndarray):  # -> np.ndarray:
        classifier = []
        y_classifier = self.models["classifier"].predict(data)
        arr = []
        for i, item in enumerate(y_classifier):
            y_classifier = self.models["classifier"].get_snippet(item)
            X_predict = np.stack([np.append(data[i], [0]), y_classifier])
            X_predict = X_predict.reshape(self.params.size_subsequent, 2)
            arr.append(X_predict)

        predict = self.models["predictor"].predict(np.array(arr))
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
        if os.path.exists(self.params.dir_dataset + CURRENT_PARAMS_DIR):

            with open(self.params.dir_dataset + CURRENT_PARAMS_DIR) as f:
                current = json.load(f)

            for key, item in self.models.items():
                item.init_networks()
                if key not in current or not current[key]:
                    item.init_dataset()
                    result = item.train_model(self.message)
                    if result is not None:
                        fig, message = result
                        self.message(message)
                        self.bot.send_plot(fig)
                    item.del_dataset()
                else:
                    item.load_model()

    def general_test(self):
        mess = "Запуск генерального тестирования"
        self.message(mess)

        keys = list(self.models)
        keys.remove("classifier")
        keys.remove("predictor")
        self.models["clear"].init_dataset()
        test_dataset = self.models["clear"].dataset
        result = {}
        fig, ax = plt.subplots(figsize=(5, 3))
        for i in keys:
            y_predict_clear = self.models[i].predict(test_dataset.X_test)
            result["mse {0}".format(i)] = metrics.mean_squared_error(y_true=test_dataset.y_test,
                                                                     y_pred=y_predict_clear)

            result["rmse {0}".format(i)] = metrics.mean_squared_error(y_true=test_dataset.y_test,
                                                                      y_pred=y_predict_clear) * 0.5

            ax.plot(y_predict_clear,
                    label=i)
        y_predict, y_classifier = self.predict(test_dataset.X_test)

        result["mse Predictor"] = metrics.mean_squared_error(test_dataset.y_test,
                                                             y_pred=y_predict)
        result["rmse Predictor"] = metrics.mean_squared_error(test_dataset.y_test,
                                                              y_pred=y_predict) * 0.5

        plt.plot(test_dataset.y_test,
                 label="true point",
                 linestyle=":",
                 marker="x")
        plt.plot(y_predict,
                 label="snippet point",
                 linestyle="-",
                 marker="o")
        ax.legend()
        fig.savefig(self.params.dir_dataset + '/result/general_test.png')
        self.message(result)
        self.bot.send_plot(fig)
        with open(self.params.dir_dataset + "/result/general_result.json", 'w') as outfile:
            json.dump(result, outfile)
        plt.clf()
        self.message("Провел генеральное тестирование")

    def test(self):
        self.message("Тестирование")
        for key, item in self.models.items():
            item.init_dataset()
            item.test(send_message=self.bot.send_message)
            item.del_dataset()
        self.general_test()


def start(dir_, bot=None):
    start_time = time.time()

    params = Params(dir_)
    init_time = time.time()
    center = Center(params, bot)
    print("Время инициализации модели %s" % (time.time() - init_time))
    train_time = time.time()

    center.train_model()
    print("Время обучения модели %s" % (time.time() - train_time))
    test_time = time.time()
    center.test()
    print("Время тестирования модели %s" % (time.time() - test_time))

    print("Общие время работы %s" % (time.time() - start_time))
