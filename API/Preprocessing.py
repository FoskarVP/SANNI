# Normalize,aug,search
import zipfile

import numpy as np
import pandas as pd
import random
from pathlib import Path
from matrixprofile.discover import snippets
from sklearn import preprocessing
import json
import os
import tensorflow as tf

from API import Image


def search_snippet(data: np.ndarray, fraction: float, size_subsequent: int) -> pd.DataFrame:
    """
    Поиск снипетов
    :param data: Директория временного ряда: str
    :param fraction: float
    :param size_subsequent: Размер подпоследовательности - int
    :return: Массив снипеетов - np.ndarray
    """

    snp = snippets(data,
                   num_snippets=int(data.shape[0] / size_subsequent * fraction),
                   snippet_size=size_subsequent)

    arr_snp = []
    for i, item in enumerate(snp):
        dict_ = {"key": i,
                 "snippet": item['snippet'],
                 "fraction": item['fraction']}
        neighbors = []
        for neighbor in item['neighbors']:
            neighbors.append(data[neighbor:neighbor + size_subsequent].tolist())
        dict_["neighbors"] = neighbors
        arr_snp.append(dict_)

    df = pd.DataFrame(arr_snp, columns=arr_snp[0].keys())
    df = augmentation(df)
    return df


def normalize(sequent: np.ndarray) -> np.ndarray:
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(sequent.reshape(-1, 1)).T[0]
    return x_scaled


def augmentation(data: pd.DataFrame, e=0.01):
    """
    Увеличение и балансировка соседей.
    Все соседи сниппетов увеличиваются до количеста соседей у сниппета с максиальным fraction
    :param data: dataframe, в котором хранятся сниппеты и их соседи
    :param e: 0<e<1 процент, на который можно сдвинуть точку
    :return: возвращается dataframe той же структуры, но со сбалансированными соседями
    """
    subseq_count = [(i, len(np.array(data.neighbors.iloc[i]))) for i in range(0, len(data.neighbors))]
    max_subseq_count = max([subseq_count[i][1] for i in range(0, len(subseq_count))])

    new_neighbors_all = []
    for cl in range(0, len(data.neighbors)):
        if subseq_count[cl][1] == max_subseq_count:
            new_neighbors_all.append(data.neighbors[cl].copy())
            continue
        neighbors = data.neighbors[cl].copy()
        need_new_neighbors = (max_subseq_count - subseq_count[cl][1])
        need_double_new = need_new_neighbors - subseq_count[cl][1] if need_new_neighbors - subseq_count[cl][
            1] > 0 else 0
        need_new_neighbors -= need_double_new
        for i in range(0, need_new_neighbors):
            new_neighbor = neighbors[i]
            new_neighbor[random.randint(0, len(neighbors[i]) - 1)] *= 1 + random.uniform(-e, e)
            neighbors.append(new_neighbor)
            if need_double_new > 0:
                new_neighbor = neighbors[i]
                new_neighbor[random.randint(0, len(neighbors[i]) - 1)] *= 1 + random.uniform(-e, e)
                neighbors.append(new_neighbor)
                need_double_new -= 1
        new_neighbors_all.append(neighbors)

    data['neighbors'] = new_neighbors_all
    return data


def create_dataset(size_subsequent: int, dataset: str, fraction: float):
    """
    Создает zip архив в директории датасета с размеченными датасетами
    :param size_subsequent: Размер подпоследовательности
    :param dataset: Директория датасета
    :param fraction: минимальный fraction
    """
    p = Path(dataset + "/data_origin.txt")
    data = np.loadtxt(p)
    data_norm = normalize(data)
    X = []
    y = []

    for i in range(size_subsequent - 1, len(data) - size_subsequent - 1):
        X.append(json.dumps(data_norm[i:i + size_subsequent - 1].tolist()))
        y.append(json.dumps(data_norm[i + size_subsequent - 1]))

    y = np.array(y)

    zipf = zipfile.ZipFile(dataset + '/dataset.zip', 'w', zipfile.ZIP_DEFLATED)

    filename = 'Clear'
    pd.DataFrame({"X": X, "y": y}).to_csv(f'{filename}.csv')
    zipf.write(filename + '.csv')
    os.remove(filename + '.csv')
    del X, y
    snippet_list = search_snippet(data=data_norm,
                                  fraction=fraction,
                                  size_subsequent=size_subsequent)

    del data_norm
    snippet_save = snippet_list.copy()
    print("Найденно снипеттов: ", snippet_list.shape[0])
    snippet_save.neighbors = snippet_save.neighbors.apply(lambda x: json.dumps(x))
    snippet_save.snippet = snippet_save.snippet.apply(lambda x: json.dumps(x.tolist()))

    snippet_save.to_csv(dataset + "/snippet.csv", )
    del snippet_save
    X_classifier = []
    X_predict = []
    y_predict = []
    y_classifier = []

    for i, item in snippet_list.iterrows():
        for neighbour in item.neighbors:
            if len(neighbour) == size_subsequent:
                X_classifier.append(json.dumps(Image.subsequent_to_image(neighbour[:-1]).tolist()))
                X_predict.append(json.dumps(np.stack([np.append(neighbour[:-1], [0]), item.snippet]).tolist()))
                y_classifier.append(item["key"])
                y_predict.append(neighbour[-1])
        del item
    del snippet_list
    y_classifier = tf.keras.utils.to_categorical(np.array(y_classifier))
    filename = 'Classifier'
    pd.DataFrame({"X": X_classifier, "y": y_classifier.tolist()}) \
        .to_csv(f'{filename}.csv')
    zipf.write(filename + '.csv')
    os.remove(filename + '.csv')

    filename = 'Predict'
    pd.DataFrame({"X": X_predict, "y": y_predict}, columns=["X", "y"]) \
        .to_csv(f'{filename}.csv')
    zipf.write(filename + '.csv')
    os.remove(filename + '.csv')
    zipf.close()
    result = {
        "size_subsequent": size_subsequent,
        "classifier": False,
        "predict": False,
        "clear": False,
        "fraction": fraction
    }

    with open(dataset + '\current_params.json', 'w') as outfile:
        json.dump(result, outfile)
    print("Создал датасет")
