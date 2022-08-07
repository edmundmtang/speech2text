import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import multiprocessing as mp
import torch
import time

def recorder(x):
    num = 10

    print(num*x)


if __name__ == '__main__':
    

    p = mp.Process(target=recorder, args=(11,))
    p.start()
    p.join()
