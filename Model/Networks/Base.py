from abc import ABC, abstractmethod
from Model.Dataset import DataSet
import numpy as np
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import json
import os
from sklearn import metrics


class BaseModel:

    def __init__(self, size_subsequent: int,
                 dataset: str,
                 load=None) -> None:
        """
        Инициализациия модели
        :param size_subsequent: размер подпоследовательности - int
        :param dataset: путь к датасету - str
        :param load: Путь к файлу модели = str
        """
        self.dir_dataset = dataset
        self.name = "Base"
        self.epochs = 0
        self.bath_size = 0
        self.pool = 0
        self.dataset = DataSet()
        self.size_subsequent = size_subsequent
        self.model = Sequential()
        self.dataset_shuffle = False
        pass

    def init_networks(self):
        print("Инициализация сети")

    def load_model(self):
        """
        Загрузка модели
        """
        self.model = load_model(self.dir_dataset + "/networks/{0}.h5".format(self.name))
        print("Загрузка {0} сети из файла".format(self.name))
        pass

    def init_dataset(self):
        self.dataset = DataSet(self.dir_dataset, self.bath_size, name=self.name)

    def train_model(self, send_message=print):
        send_message("Запуск обучения {0}".format(self.name))
        history = self.model.fit(self.dataset.X_train,
                                 self.dataset.y_train,
                                 validation_data=(self.dataset.X_valid,
                                                  self.dataset.y_valid),
                                 batch_size=self.bath_size,
                                 epochs=self.epochs)

        fig, ax = plt.subplots(figsize=(5, 3))
        message = {}
        for key, item in history.history.items():
            ax.plot(item, label=key)
            message[key] = item[-1]
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        plt.savefig(self.dir_dataset + '/result/{0}.png'.format(self.name))
        self.save_model()
        return fig, message

    def save_model(self) -> None:
        """
        Сохранения модели в файл
        """
        if not os.path.exists(self.dir_dataset + "/networks"):
            os.mkdir(self.dir_dataset + "/networks")

        self.model.save(self.dir_dataset + "/networks/{0}.h5".format(self.name))

        with open(self.dir_dataset + "/current_params.json") as f:
            current = json.load(f)
        current[self.name] = True
        with open(self.dir_dataset + '/current_params.json', 'w') as outfile:
            json.dump(current, outfile)
        print("Сохранил модель")
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Сохранения модели в файл
        :param data: Входная последовательномть - np.ndarray
        :return: массив ответов
        """
        return self.model.predict(data)

    def test(self, send_message=print):
        y_predict = self.predict(self.dataset.X_test)

        result = {
            "mse": metrics.mean_squared_error(y_true=self.dataset.y_test, y_pred=y_predict),
            "rmse": metrics.mean_squared_error(y_true=self.dataset.y_test, y_pred=y_predict) * 0.5
        }

        with open(self.dir_dataset + "/result/{0}_result.txt".format(self.name), 'w') as outfile:
            json.dump(result, outfile)

        send_message("Провел внутренние тестирование {0}\n{1}".format(self.name, result))
        pass

    def del_dataset(self):
        del self.dataset
        pass
