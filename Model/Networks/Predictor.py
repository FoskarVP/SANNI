from Model.Networks.Base import BaseModel
from Model.Dataset import DataSet

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from keras.layers import Conv2D
from keras.models import Model, Input, load_model
from keras.layers import Dense
from tensorflow.keras.layers import GRU, Dropout


class Predictor(BaseModel):
    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.bath_size = 25
        self.epochs = 60
        self.loss = "mse"
        self.optimizer = "adam"
        self.layers = [128]
        self.dataset = DataSet(dataset, self.bath_size, name="Predict",shuffle=True)
        self.model = self.__init_networks()
        print("Инициализации сверточной сети")

    def __init_networks(self):
        input_layer = Input((self.size_subsequent, 2),
                            name="img_input",
                            dtype='float32')
        output = input_layer

        ## ведутся работы
        for i in self.layers[:-1]:
            output = Conv2D(i[0], (i[1], i[1]), kernel_initializer='he_normal', activation='relu')(output)

        output = GRU(self.layers[-1],
                     kernel_initializer='he_normal',
                     activation='relu')(output)
        output = Dropout(0.2)(output)
        output = Dense(1)(output)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        return model

    def train_model(self):
        print("Запуск обучения Предсказателя")

        history = self.model.fit(self.dataset.X_train.
                                 reshape(self.dataset.X_train.shape[0],
                                         self.dataset.X_train.shape[2],
                                         self.dataset.X_train.shape[1]),
                                 self.dataset.y_train,
                                 validation_data=(self.dataset.X_valid.
                                                  reshape(self.dataset.X_valid.shape[0],
                                                          self.dataset.X_valid.shape[2],
                                                          self.dataset.X_valid.shape[1]),
                                                  self.dataset.y_valid),
                                 batch_size=self.bath_size, epochs=self.epochs)

        plt.plot(history.history["loss"], label="train_dataset")
        plt.plot(history.history["val_loss"], label="valid_dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(self.dir_dataset + '/result/Predictor.png')
        print("Провел обучение")
        self.save_model()

    def load_model(self):
        self.model = load_model(self.dir_dataset + "/networks/predict.h5")
        print("Загрузка предсказателя сети из файла")

    def save_model(self) -> None:

        if not os.path.exists(self.dir_dataset + "/networks"):
            os.mkdir(self.dir_dataset + "/networks")

        self.model.save(self.dir_dataset + "/networks/predict.h5")

        with open(self.dir_dataset + "/current_params.json") as f:
            current = json.load(f)
        current["predict"] = True
        with open(self.dir_dataset + '/current_params.json', 'w') as outfile:
            json.dump(current, outfile)
        print("Сохранил модель")

    def test(self):
        y_predict = self.predict(self.dataset.X_test.reshape(self.dataset.X_test.shape[0],
                                                             self.dataset.X_test.shape[2],
                                                             self.dataset.X_test.shape[1]))

        print("mse предсказателя - {0};".
              format(metrics.mean_squared_error(y_true=self.dataset.y_test,
                                                y_pred=y_predict)))
        print("rmse предсказателя- {0};".
              format(metrics.mean_squared_error(y_true=self.dataset.y_test,
                                                y_pred=y_predict) * 0.5))
        result = {
            "mse": metrics.mean_squared_error(y_true=self.dataset.y_test, y_pred=y_predict),
            "rmse": metrics.mean_squared_error(y_true=self.dataset.y_test, y_pred=y_predict) * 0.5
        }
        with open(self.dir_dataset + "/result/predictor_result.txt", 'w') as outfile:
            json.dump(result, outfile)

        print("Провел внутренние тестирование предсказателя на основе сниппетов")
