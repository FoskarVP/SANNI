import sys
import json
from Head.Params import Params
from Head.Center import Center

if len(sys.argv) != 2:
    print("ожидается 1 аргумента")
    sys.exit()

import time

start_time = time.time()

params = Params(sys.argv[1])
init_time = time.time()
center = Center(params)
print("Время инициализации модели %s" % (time.time() - init_time))
train_time = time.time()

center.train_model()
print("Время обучения модели %s" % (time.time() - train_time))
test_time = time.time()
center.test()
print("Время тестирования модели %s" % (time.time() - test_time))

print("Общие время работы %s" % (time.time() - start_time))
