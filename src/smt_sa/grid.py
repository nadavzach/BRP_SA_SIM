from smt_sa.node import NodePU, NodeMem


BUF_A = 0
BUF_B = 1


class Grid:
    def __init__(self, dim, threads, max_depth=4096):
        self.mem_a = [NodeMem('mem_a_{}'.format(i), threads) for i in range(dim)]
        self.mem_b = [NodeMem('mem_b_{}'.format(i), threads) for i in range(dim)]
        self.nodes = [[NodePU('node_{}_{}'.format(i, j), threads, max_depth) for j in range(dim)] for i in range(dim)]

        self._dim = dim
        self._threads = threads

        # Connect PUs
        for i in range(dim):
            for j in range(dim):
                if j+1 < dim:
                    self.nodes[i][j].out_a = self.nodes[i][j + 1]
                if i+1 < dim:
                    self.nodes[i][j].out_b = self.nodes[i + 1][j]

        # Connect memory units
        for i in range(dim):
            self.mem_a[i].out = self.nodes[i][0]
            self.mem_a[i].out_buf_idx = BUF_A

            self.mem_b[i].out = self.nodes[0][i]
            self.mem_b[i].out_buf_idx = BUF_B

    def push(self, a, b, thread, pad=False):
        a_H, a_W = a.shape
        b_H, b_W = b.shape
        assert(a_W == b_H)

        # Push matrix A
        for i in range(a_H):
            buf = self.mem_a[i]._buf[thread]

            if pad is True:
                for k in range(i):
                    buf._phy_fifo.append(None)
                    buf._arch_fifo.append(None)
                    buf._arch_fifo_size += 1
                    buf._phy_fifo_size += 1

            for j in range(a_W):
                buf._phy_fifo.append(a[i][j])
                buf._arch_fifo.append(a[i][j])
                buf._arch_fifo_size += 1
                buf._phy_fifo_size += 1

        for j in range(b_W):
            buf = self.mem_b[j]._buf[thread]

            if pad is True:
                for k in range(j):
                    buf._phy_fifo.append(None)
                    buf._arch_fifo.append(None)
                    buf._arch_fifo_size += 1
                    buf._phy_fifo_size += 1

            for i in range(b_H):
                buf._phy_fifo.append(b[i][j])
                buf._arch_fifo.append(b[i][j])
                buf._arch_fifo_size += 1
                buf._phy_fifo_size += 1

    def cycle(self):
        # PU computation and cycle
        for i in reversed(range(self._dim)):
            for j in reversed(range(self._dim)):
                self.nodes[i][j].do()

                for t in range(self._threads):
                    self.nodes[i][j]._buf[BUF_A][t].cycle()
                    self.nodes[i][j]._buf[BUF_B][t].cycle()

        # Memory nodes computation and cycle
        for i in reversed(range(self._dim)):
            self.mem_a[i].do()
            self.mem_b[i].do()

            for t in range(self._threads):
                self.mem_a[i]._buf[t].cycle()
                self.mem_b[i]._buf[t].cycle()

        #print(self.get_util_rate())

    def get_util_rate(self):
        util_sum = 0

        for i in range(self._dim):
            for j in range(self._dim):
                util_sum += self.nodes[i][j].is_util()

        return util_sum / (self._dim * self._dim)
