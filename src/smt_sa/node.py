from smt_sa.fifo import FIFO
import numpy as np


BUF_A = 0
BUF_B = 1


class NodePU:
    def __init__(self, name: str, threads: int, max_depth=4096):
        self.name = name
        self.out_a = None   # type: NodePU
        self.out_b = None   # type: NodePU

        self._threads = threads
        self._buf = {BUF_A: [FIFO('{}_a_fifo'.format(name)) for i in range(threads)],
                     BUF_B: [FIFO('{}_b_fifo'.format(name)) for i in range(threads)]}
        self._max_depth = max_depth

        self._acc = 0
        self._acc_t = np.zeros(threads, dtype=int)
        self._halt = [False for t in range(threads)]

        self._utilized = False

    def push(self, x, buf, thread):
        self._buf[buf][thread].push(x)

    def pop(self, buf, thread):
        return self._buf[buf][thread].pop()

    def is_valid(self, thread):
        if self._buf[BUF_A][thread].size() == 0 or self._buf[BUF_B][thread].size() == 0:
            return False
        else:
            return True

    def is_ready(self, buf, thread):
        if self._buf[buf][thread].size() < self._max_depth:
            return True
        else:
            return False

    def is_ready_out(self, thread):
        a_out_ready = True if self.out_a is None else self.out_a.is_ready(BUF_A, thread)
        b_out_ready = True if self.out_b is None else self.out_b.is_ready(BUF_B, thread)

        return a_out_ready and b_out_ready

    def do(self):
        self._utilized = False

        for t in range(self._threads):
            if self.is_valid(t) and not self.is_halt(t) and self.is_ready_out(t):
                a = self.pop(BUF_A, t)
                b = self.pop(BUF_B, t)

                self._acc_t[t] += 1

                if self.out_a is not None:
                    self.out_a.push(a, BUF_A, t)
                if self.out_b is not None:
                    self.out_b.push(b, BUF_B, t)

                # Zero-bypass
                if a != 0 and b != 0:
                    self._acc += a * b
                    self._utilized = True
                    break

    def get_acc(self):
        return self._acc

    def get_acc_t(self, thread):
        return self._acc_t[thread]

    def reset_acc(self):
        self._acc = 0

    def reset_acc_t(self, thread=None):
        if thread is None:
            for t in range(self._threads):
                self._acc_t[t] = 0
        else:
            self._acc_t[thread] = 0

    def is_halt(self, thread):
        return self._halt[thread]

    def halt(self, thread):
        self._halt[thread] = True

    def release(self, thread=None):
        if thread is None:
            for t in range(self._threads):
                self._halt[t] = False
        else:
            self._halt[thread] = False

    def is_util(self):
        return self._utilized


class NodeMem:
    def __init__(self, name: str, threads: int):
        self._name = name
        self.out = None   # type: NodePU
        self.out_buf_idx = None

        self._thread = threads
        self._buf = [FIFO('{}_fifo'.format(name)) for i in range(threads)]

    def push(self, x, thread):
        self._buf[thread].push(x)

    def pop(self, thread):
        return self._buf[thread].pop()

    def do(self):
        for t in range(self._thread):
            if self._buf[t].size() > 0:
                val = self.pop(t)

                if val is not None:
                    self.out.push(val, self.out_buf_idx, t)
