import time
import random
import multiprocessing
from contextlib import contextmanager


class WorkerQueue:

    def __init__(self,num_workers):
        self.queue = multiprocessing.Manager().Queue()
        _ = [ self.queue.put(i) for i in range(num_workers)]

    @contextmanager
    def next_available(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


class Parallel:

    """ Turn  f(worker, *args, **kwargs) into  g(*args, **kwargs)
        So if you need to call g() a lot, write a version with one pre-pended argument then do g = Parallel(f)
    """

    def __init__(self, func, num_workers):
        self.queue = WorkerQueue(num_workers=num_workers)
        self.func = func

    def __call__(self, *args, **kwargs):
        with self.queue.next_available() as worker:
            return self.func(worker,*args,**kwargs)


# Usage example....
if __name__=='__main__':

    def boss(i, x):
        print('Gonna send '+str(x)+' to server '+str(i))
        time.sleep(random.choice(range(10-i)))
        return x

    task = Parallel(boss, num_workers=5)

    from multiprocessing import Pool
    with Pool(5) as p:
        print( p.map(task,[1,3,4,2,3,1,7,3,2,1,4,1] ) )