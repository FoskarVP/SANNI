if __name__ == '__main__':
    import sys
    import json
    from Head.Params import Params
    from Head.Center import start

    print("Привет")
    if len(sys.argv) != 2:
        print("ожидается 1 аргумента")
        sys.exit()

    import time
    start(sys.argv[1])
