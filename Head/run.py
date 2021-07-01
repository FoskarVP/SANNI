import sys
import json
from Head.Params import Params
from Head.Center import Center

if len(sys.argv) != 2:
    print("ожидается 1 аргумента")
    sys.exit()

params = Params(sys.argv[1])

center = Center(params)

center.train_model()
center.test()
