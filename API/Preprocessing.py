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


def search_snippet(data: np.ndarray, snippet_count: int, size_subsequent: int) -> pd.DataFrame:
    """
    Поиск снипетов
    :param data: Директория временного ряда: str
    :param snippet_count: int
    :param size_subsequent: Размер подпоследовательности - int
    :return: Массив снипеетов - np.ndarray
    """

    snp = snippets(data,
                   num_snippets=snippet_count,
                   snippet_size=size_subsequent)

    arr_snp = []
    for i, item in enumerate(snp):
        dict_ = {"key": i,
                 "snippet": item['snippet'],
                 "fraction": item['fraction']}
        neighbors = []
        index = []
        for neighbor in item['neighbors']:
            neighbors.append(data[neighbor:neighbor + size_subsequent].tolist())
            index.append(neighbor)
        dict_["neighbors"] = neighbors
        dict_["neighbors_index"] = index
        arr_snp.append(dict_)
        del item

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


def create_dataset(size_subsequent: int, dataset: str, snippet_count: int) -> int:
    """
    Создает zip архив в директории датасета с размеченными датасетами
    :param size_subsequent: Размер подпоследовательности
    :param dataset: Директория датасета
    :param snippet_count: минимальный fraction
    :return Возращает колличество сниппетов
    """
    if not os.path.isdir("{0}/dataset".format(dataset)):
        os.mkdir("{0}/dataset".format(dataset))
    p = Path(dataset + "/data_origin.txt")
    data = np.loadtxt(p)
    data_norm = normalize(data)
    dataset = "{0}/dataset".format(dataset)
    X = []
    y = []

    for i in range(0, len(data) - size_subsequent - 1):
        X.append(json.dumps(data_norm[i:i + size_subsequent - 1].tolist()))
        y.append(json.dumps(data_norm[i + size_subsequent - 1]))

    y = np.array(y)

    print("создал архив")

    filename = 'clear'
    pd.DataFrame({"X": X, "y": y}).to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')
    del X
    print("Начал поиск сниппетов")
    snippet_list = search_snippet(data=data_norm,
                                  snippet_count=snippet_count,
                                  size_subsequent=size_subsequent)
    count_snippet = snippet_list.shape[0]
    print("Найденно снипеттов: ", count_snippet)

    X_classifier = []
    y_classifier = []

    for i, item in snippet_list.iterrows():
        for neighbour in item.neighbors:
            if len(neighbour) == size_subsequent:
                X_classifier.append(json.dumps(np.array(neighbour[:-1]).tolist()))
                y_classifier.append(item["key"])
    print("Создал датасет классификатора")

    y_classifier = tf.keras.utils.to_categorical(np.array(y_classifier))
    filename = 'classifier'
    pd.DataFrame({"X": X_classifier, "y": y_classifier.tolist()}) \
        .to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')
    del X_classifier, y_classifier

    X_predict = []
    y_predict = []
    for i, item in snippet_list.iterrows():
        for neighbour in item.neighbors:
            if len(neighbour) == size_subsequent:
                X_predict.append(json.dumps(np.stack([np.append(neighbour[:-1], [0]), item.snippet]).tolist()))
                y_predict.append(neighbour[-1])

    filename = 'predictor'
    pd.DataFrame({"X": X_predict, "y": y_predict}, columns=["X", "y"]) \
        .to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')

    X_predict = []
    y_predict = []

    for i in range(size_subsequent - 1, len(data) - size_subsequent - 1):
        subsequent = data_norm[i:i + size_subsequent - 1].tolist()
        number = 1
        for j, item in snippet_list.iterrows():
            if i in item.neighbors_index:
                number = i
                break
        X_predict.append(json.dumps(np.stack([np.array(subsequent),
                                              np.full(size_subsequent - 1, number)]).tolist()))
        y_predict.append(json.dumps(data_norm[i + size_subsequent - 1]))


    print("Создал датасет предсказателя")
    snippet_list.neighbors = snippet_list.neighbors.apply(lambda x: json.dumps(x))
    snippet_list.snippet = snippet_list.snippet.apply(lambda x: json.dumps(x.tolist()))
    snippet_list.to_csv(dataset + "/snippet.csv", )

    del snippet_list
    filename = 'predictor_label'
    pd.DataFrame({"X": X_predict, "y": y_predict}, columns=["X", "y"]) \
        .to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')

    result = {
        "size_subsequent": size_subsequent,
        "classifier": False,
        "predictor": False,
        "predictor_label": False,
        "clear": False,
        "snippet_count": snippet_count
    }

    with open('{0}\\current_params.json'.format(dataset), 'w') as outfile:
        json.dump(result, outfile)

    print("Создал датасет")
    return count_snippet
