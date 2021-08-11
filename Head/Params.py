import json
import os

class Params:
    def __init__(self,
                 _dir: str):
        print("Текущая деректория:", os.getcwd())
        print(_dir + "\\params.json")
        with open(_dir + "\\params.json") as f:
            params = json.load(f)
        self.dir_dataset = _dir
        self.snippet_count = params["snippet_count"]
        self.percent_test = params["percent_test"]
        self.size_subsequent = params["size_subsequent"]
        self.dataset_name = params["dataset_name"]