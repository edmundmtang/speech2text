def foo(x):
    return x * x

print(foo(3))

class bar(object):
    def __init__(self,foo):
        self.foo = foo

baka = bar(foo)

import multiprocessing as mp

mgr = mp.Manager()
ns = mgr.Namespace()
ns.baka = baka

