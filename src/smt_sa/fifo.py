from collections import deque


class FIFO:
    def __init__(self, name):
        self.name = name

        self._arch_fifo = deque()
        self._phy_fifo = deque()

        self._pushed = False
        self._popped = False
        self._last_pushed = None
        self._arch_fifo_size = 0
        self._phy_fifo_size = 0

    def peek(self, pos=0):
        if self._arch_fifo.__len__() == 0:
            return None
        else:
            return self._arch_fifo[pos]

    def push(self, x):
        if self._pushed:
            raise ValueError("Can't push to the queue more than once per cycle")

        self._phy_fifo.append(x)
        self._pushed = True
        self._last_pushed = x
        self._phy_fifo_size += 1

    def pop(self):
        if self._popped:
            raise RuntimeError("Can't pop from the queue more than once per cycle")

        if self._phy_fifo_size > 0:
            self._phy_fifo.popleft()
            self._phy_fifo_size -= 1

        self._popped = True
        return self.peek()

    def size(self):
        return self._arch_fifo_size

    def cycle(self):
        if self._pushed:
            self._arch_fifo.append(self._last_pushed)
            self._pushed = False
            self._last_pushed = None

        if self._popped:
            self._arch_fifo.popleft()
            self._popped = False

        self._arch_fifo_size = self._phy_fifo_size
