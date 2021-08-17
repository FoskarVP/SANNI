from Model.Dataset import DataSet
from Model.Networks.Base import BaseModel
from pathlib import Path
import numpy as np


class Func(BaseModel):
    def init_dataset(self):
        self.dataset = DataSet(self.dir_dataset,
                               self.bath_size,
                               name="clear")

    def train_model(self, send_message=print):
        print("Запуск обучения: ",self.name)