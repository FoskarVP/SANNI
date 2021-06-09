import numpy as np


class Snippet:
    def __init__(self,
                 fraction: float,
                 index: int,
                 index_neighbors: np.ndarray,
                 data: np.ndarray):

        self.fraction = fraction
        self.index = index
        self.index_neighbors = index_neighbors
        self.data = data
