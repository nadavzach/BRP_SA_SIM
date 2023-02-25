from smt_sa.smt_sa import SMTSA
import math
import numpy as np
import concurrent.futures
import progressbar
from collections import deque
import time
import cpy_smt_sa


class ParallelSMTSA:
    def __init__(self, dim, threads: int, max_depth=4096):
        self._dim = dim
        self._threads = threads
        self._depth = max_depth

        self._a = None
        self._b = None

    def set_inputs(self, a, b):
        assert(len(a.shape) == 3)   # a is 3 dimensions (batch, height, width)
        assert(len(b.shape) == 2)   # b is 2 dimensions (height, width)
        assert(a.shape[2] == b.shape[0])

        self._a = a
        self._b = b

    def _go(self, a_crop):
        return cpy_smt_sa.run(self._dim, self._threads, self._depth, a_crop, self._b)

    def go(self, max_workers=16):
        out_unf = np.zeros((self._a.shape[0], self._a.shape[1], self._b.shape[1]))
        threads = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for b in range(self._a.shape[0]):
                a_crop = self._a[b]
                a_crop = a_crop[np.newaxis, :, :]
                threads[executor.submit(self._go, a_crop)] = '{}'.format(b)

            for counter, thread in enumerate(concurrent.futures.as_completed(threads)):
                try:
                    result_batch = int(threads[thread])
                    out_unf[result_batch, :, :] = thread.result()
                except Exception as exc:
                    print(exc)

        return out_unf
