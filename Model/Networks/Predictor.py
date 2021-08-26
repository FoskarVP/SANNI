from Model.Networks.Base import BaseModel
import json
import tensorflow.keras as ks
from tensorflow.keras import Input, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU, Dropout
from keras.regularizers import l1, l2, l1_l2


class Predictor(BaseModel):
    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.bath_size = 25
        self.epochs = 100
        self.name = "predictor"
        self.loss = "mse"
        self.optimizer = optimizers.Adam(learning_rate=0.0005)
        self.input = (self.size_subsequent, 2)
        try:
            with open(dataset + "/networks.json", "r") as read_file:
                self.layers = json.load(read_file)[self.name]
        except Exception as e:
            print(e)
            print(123)
            self.layers = [128]

    def init_networks(self):
        input_layer = Input(self.input,
                            name="img_input",
                            dtype='float32')
        output = input_layer

        for i in self.layers[:-1]:
            output = GRU(i,
                         return_sequences=True,
                         kernel_initializer='he_normal',
                         #          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                         #           bias_regularizer=l2(1e-4),
                         #           activity_regularizer=l2(1e-5),

                         activation='relu')(output)
            output = Dropout(0.05)(output)
        output = GRU(self.layers[-1],
                     kernel_initializer='he_normal',
                     #    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                     #    bias_regularizer=l2(1e-4),
                     #    activity_regularizer=l2(1e-5),
                     activation='relu')(output)
        output = Dense(1)(output)
        model = Model(inputs=input_layer, outputs=output, name=self.name)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        self.model = model

    def init_dataset(self):
        super().init_dataset()
        if len(self.dataset.X_train.shape) > 2:
            self.dataset.X_train = self.dataset.X_train. \
                reshape(self.dataset.X_train.shape[0],
                        self.dataset.X_train.shape[2],
                        self.dataset.X_train.shape[1])
            self.dataset.X_valid = self.dataset.X_valid. \
                reshape(self.dataset.X_valid.shape[0],
                        self.dataset.X_valid.shape[2],
                        self.dataset.X_valid.shape[1])
            self.dataset.X_test = self.dataset.X_test. \
                reshape(self.dataset.X_test.shape[0],
                        self.dataset.X_test.shape[2],
                        self.dataset.X_test.shape[1])


class Predictor_label(Predictor):
    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.name = "predictor_label"
        self.input = (self.size_subsequent - 1, 2)
        self.optimizer = ks.optimizers.Adam(learning_rate=0.05)
        try:
            with open(dataset + "/networks.json", "r") as read_file:
                self.layers = json.load(read_file)[self.name]
        except Exception as e:
            print(e)
            print(123)
            self.layers = [128]
