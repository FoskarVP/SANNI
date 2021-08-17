from Model.Networks.Predictor import Predictor


class Clear(Predictor):
    def __init__(self, size_subsequent: int, dataset: str, load=None) -> None:
        super().__init__(size_subsequent, dataset, load)
        self.name = "clear"
        self.input = (self.size_subsequent - 1, 1)





