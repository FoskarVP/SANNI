# Normalize,aug,search
import numpy as np
import pandas as pd
import random


def search_snippet(data: str, fraction: float, size_subsequent: int) -> np.ndarray:
    """
    Поиск снипетов
    :param data: Директория временного ряда: str
    :param fraction: float
    :param size_subsequent: Размер подпоследовательности - int
    :return: Массив снипеетов - np.ndarray
    """
    return np.array([])


def normalize(subsequent: np.ndarray) -> np.ndarray:
    return np.array([])


def augmentation(data: pd.DataFrame, e = 0.01):
    subseq_count = [(i, len(np.array(data.neighbors.iloc[i]))) for i in range(0, len(data.neighbors))]
    max_subseq_count = max([subseq_count[i][1] for i in range(0, len(subseq_count))])

    new_neighbors_all = []
    for cl in range(0, len(data.neighbors)):
        if subseq_count[cl][1] == max_subseq_count:
            new_neighbors_all.append(data.neighbors[cl].copy())
            continue
        neighbors = data.neighbors[cl].copy()
        need_new_neighbors = (max_subseq_count - subseq_count[cl][1])
        need_double_new = need_new_neighbors-subseq_count[cl][1] if need_new_neighbors-subseq_count[cl][1] > 0 else 0
        need_new_neighbors -= need_double_new
        for i in range(0, need_new_neighbors):
            new_neighbor = neighbors[i]
            new_neighbor[random.randint(0, len(neighbors[i])-1)] *= 1+random.uniform(-e, e)
            neighbors.append(new_neighbor)
            if need_double_new > 0 :
                new_neighbor = neighbors[i]
                new_neighbor[random.randint(0, len(neighbors[i])-1)] *= 1+random.uniform(-e, e)
                neighbors.append(new_neighbor)
                need_double_new -= 1
        new_neighbors_all.append(neighbors)
    return np.array(new_neighbors_all)


