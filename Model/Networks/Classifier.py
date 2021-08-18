from Model.Networks.Base import BaseModel
from Model.Dataset import DataSet

import os

import numpy as np
import json
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Conv1D, Activation
import tensorflow as tf
from keras.models import Model, Input, load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, AveragePooling2D, AveragePooling1D, GlobalAvgPool1D


class Classifier(BaseModel):
    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super(Classifier, self).__init__(size_subsequent, dataset, load)

        # засунуть в абстрактный класс
        self.bath_size = 25
        self.name = "classifier"
        self.pool = 2
        self.dataset_shuffle = True
        try:
            with open(dataset + "/networks.json", "r") as read_file:
                self.layers = json.load(read_file)["classifier"]
        except BaseException:
            self.layers = [[128, 5], [128, 5], [128, 5]]
        self.epochs = 60
        self.optimizer = "adam"
        self.loss = "categorical_crossentropy"
        self.metrics = [tf.keras.metrics.Precision()]

        self.snippet_list = pd.read_csv(self.dir_dataset + "/snippet.csv",
                                        converters={"snippet": json.loads}).snippet.values

    def init_networks(self):
        input_layer = Input((self.size_subsequent - 1, 1),
                            name="img_input",
                            dtype='float32')
        output = input_layer
        for i in self.layers[:-1]:
            output = Conv1D(i[0], i[1], kernel_initializer='he_normal', activation='relu')(output)
            if len(i) == 3:
                if i[2] == "a":
                    output = AveragePooling1D(pool_size=2)(output)
            else:
                output = AveragePooling1D(pool_size=2)(output)
            output = Dropout(0.05)(output)

        output = Conv1D(self.layers[-1][0], self.layers[-1][1],
                        kernel_initializer='he_normal', activation='relu')(output)
        output = GlobalAvgPool1D()(output)
        output = Dropout(0.25)(output)
        output = Dense(self.snippet_list.shape[0])(output)
        y_pred = Activation('softmax', name='softmax')(output)

        model = Model(inputs=input_layer, outputs=y_pred)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        model.summary()
        self.model = model

    def get_snippet(self, class_snip: int) -> np.ndarray:
        return self.snippet_list[class_snip]

    def predict(self, data: np.ndarray):
        """
        Сохранения модели в файл
        :param data: Входная последовательномть - np.ndarray
        :return: массив ответов
        """
        data = super(Classifier, self).predict(data)
        return np.argmax(data, axis=1)

    def test(self, send_message=print):
        y_predict = self.predict(self.dataset.X_test)
        result = {
            "accuracy": metrics.accuracy_score(y_true=np.argmax(self.dataset.y_test, axis=1), y_pred=y_predict),
            "recall": metrics.recall_score(y_true=np.argmax(self.dataset.y_test, axis=1), y_pred=y_predict,
                                           average="weighted"),
            "precision": metrics.precision_score(y_true=np.argmax(self.dataset.y_test, axis=1), y_pred=y_predict,
                                                 average="weighted"),
            "f1": metrics.f1_score(y_true=np.argmax(self.dataset.y_test, axis=1), y_pred=y_predict, average="weighted"),
        }

        send_message(str(result))

        with open(self.dir_dataset + "/result/classifier_result.txt", 'w') as outfile:
            json.dump(result, outfile)

        send_message("Провел внутренние тестирование {0}\n{1}".format(self.name, result))
