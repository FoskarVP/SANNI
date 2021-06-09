class Params:
    def __init__(self,
                 dir_dataset: str,
                 fraction: float,
                 percent_test: int,
                 size_subsequent: int):
        self.dir_dataset = dir_dataset
        self.fraction = fraction
        self.percent_test = percent_test
        self.size_subsequent = size_subsequent
