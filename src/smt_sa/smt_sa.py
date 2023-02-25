import math
import numpy as np
import progressbar
import concurrent.futures
from smt_sa.grid import Grid


class SMTSA:
    def __init__(self, dim, threads: int, max_depth=4096):
        self.grid = Grid(dim, threads, max_depth)

        self._dim = dim
        self._threads = threads

        self._a = None
        self._b = None

        self.cycles = 0

    def set_inputs(self, a, b):
        assert(len(a.shape) == 3)   # a is 3 dimensions (batch, height, width)
        assert(len(b.shape) == 2)   # b is 2 dimensions (height, width)
        assert(a.shape[2] == b.shape[0])

        self._a = a
        self._b = b

    def get_tile(self, idx=None):
        a_tile_H = self._dim
        b_tile_W = self._dim

        subtile_start, subtile_end = self._subtile_dict()

        batch, i, j = idx
        tile_a = {}
        for t in range(self._threads):
            tile_a[t] = self._a[batch,
                        (i * a_tile_H):((i + 1) * a_tile_H),
                        subtile_start[t]:subtile_end[t]]

            if self._dim > tile_a[t].shape[0]:
                tile_a[t] = np.pad(tile_a[t], ((0, self._dim - tile_a[t].shape[0]), (0, 0)),
                                   mode='constant', constant_values=((0, 0), (0, 0)))

        tile_b = {}
        for t in range(self._threads):
            tile_b[t] = self._b[subtile_start[t]:subtile_end[t],
                        (j * b_tile_W):((j + 1) * b_tile_W)]

            if self._dim > tile_b[t].shape[1]:
                tile_b[t] = np.pad(tile_b[t], ((0, 0), (0, self._dim - tile_b[t].shape[1])),
                                   mode='constant', constant_values=((0, 0), (0, 0)))

        return tile_a, tile_b

    def tile_generator(self):
        a_tile_H = self._dim
        a_tile_W = self._a.shape[2]
        a_tiles = math.ceil(self._a.shape[1] / self._dim)

        b_tile_H = self._b.shape[0]
        b_tile_W = self._dim
        b_tiles = math.ceil(self._b.shape[1] / self._dim)

        # TODO: redundant, remove?
        assert(a_tile_W == b_tile_H)

        subtile_start, subtile_end = self._subtile_dict()

        for batch in range(self._a.shape[0]):
            for i in range(a_tiles):
                tile_a = {}
                for t in range(self._threads):
                    tile_a[t] = self._a[batch,
                                        (i * a_tile_H):((i + 1) * a_tile_H),
                                        subtile_start[t]:subtile_end[t]]

                    # Pad tile with zeros in case the tile is smaller than the systolic array.
                    # This is done in this stage so values will be pushed to the memory nodes, and synchronization
                    # within the systolic array will remain.
                    if self._dim > tile_a[t].shape[0]:
                        tile_a[t] = np.pad(tile_a[t], ((0, self._dim - tile_a[t].shape[0]), (0, 0)),
                                           mode='constant', constant_values=((0, 0), (0, 0)))

                tile_b = {}
                for j in range(b_tiles):
                    for t in range(self._threads):
                        tile_b[t] = self._b[subtile_start[t]:subtile_end[t],
                                            (j * b_tile_W):((j + 1) * b_tile_W)]

                        if self._dim > tile_b[t].shape[1]:
                            tile_b[t] = np.pad(tile_b[t], ((0, 0), (0, self._dim - tile_b[t].shape[1])),
                                               mode='constant', constant_values=((0, 0), (0, 0)))

                    yield (tile_a, tile_b)

    def _subtile_dict(self):
        subtile_start, subtile_end = [], []
        for t in range(self._threads):
            start, end = self._subtile_range(t)
            subtile_start.append(start)
            subtile_end.append(end)

        return (subtile_start, subtile_end)

    def _subtile_range(self, t):
        a_tile_W = self._a.shape[2]
        b_tile_H = self._b.shape[0]
        assert (a_tile_W == b_tile_H)

        subtile_size = math.floor(b_tile_H / self._threads)

        if t < self._threads - 1:
            thread_tile_start = int(t * subtile_size)
            thread_tile_end = int((t + 1) * subtile_size)
        else:
            thread_tile_start = int(t * subtile_size)
            thread_tile_end = b_tile_H

        return (thread_tile_start, thread_tile_end)

    def go(self, tile_idx=None):
        a_tiles = math.ceil(self._a.shape[1] / self._dim)
        b_tiles = math.ceil(self._b.shape[1] / self._dim)

        if tile_idx is None:
            tile_generator = self.tile_generator()
            tiles = next(tile_generator)
            array_ctrl = np.zeros((self._dim, self._dim))
        else:
            tiles = self.get_tile(tile_idx)
            array_ctrl = np.ones((self._dim, self._dim)) * ((tile_idx[0] * a_tiles * b_tiles) +
                                                            (tile_idx[1] * b_tiles) +
                                                            (tile_idx[2]))

        for t in range(self._threads):
            self.grid.push(tiles[0][t], tiles[1][t], t, pad=True)

        result = np.zeros((self._a.shape[0], self._a.shape[1], self._b.shape[1]))

        a_shape1_padded = a_tiles * self._dim
        b_shape1_padded = b_tiles * self._dim

        subtile_start, subtile_end = self._subtile_dict()

        computed = 0
        while_end = self._a.shape[0] * a_shape1_padded * b_shape1_padded if tile_idx is None else self._dim * self._dim

        #pbar = progressbar.ProgressBar(max_value=self._a.shape[0] * a_shape1_padded * b_shape1_padded)

        while computed < while_end:
            self.grid.cycle()
            self.cycles += 1

            for i in range(self._dim):
                for j in range(self._dim):

                    halt_count = 0
                    for t in range(self._threads):
                        if self.grid.nodes[i][j].is_halt(t):
                            halt_count += 1

                        acc_t = self.grid.nodes[i][j].get_acc_t(t)

                        if (acc_t == subtile_end[t] - subtile_start[t]) and not self.grid.nodes[i][j].is_halt(t):
                            self.grid.nodes[i][j].halt(t)

                    if halt_count == self._threads:
                        batch = math.floor(array_ctrl[i, j] / (a_tiles * b_tiles))
                        i_result = int(i + int((array_ctrl[i, j] % (a_tiles * b_tiles)) / b_tiles) * self._dim)
                        j_result = int(j + ((array_ctrl[i, j] % (a_tiles * b_tiles)) % b_tiles) * self._dim)

                        # Skip paddings
                        if i_result < result.shape[1] and j_result < result.shape[2]:
                            result[batch, i_result, j_result] = self.grid.nodes[i][j].get_acc()

                        array_ctrl[i, j] += 1
                        self.grid.nodes[i][j].reset_acc()
                        self.grid.nodes[i][j].reset_acc_t()
                        computed += 1

                        #pbar.update(computed)

                        self.grid.nodes[i][j].release()

            # At the end of each cycle check if data needs to be pushed into the memory nodes
            if self.grid.mem_a[0]._buf[0]._arch_fifo_size < 128 and tile_idx is None:
                try:
                    tiles = next(tile_generator)
                    for t in range(self._threads):
                        self.grid.push(tiles[0][t], tiles[1][t], t, pad=True)
                except StopIteration:
                    continue

        #result_ref = np.matmul(self._a, self._b)
        #print('Result reference =\n{}'.format(result_ref))
        #print('\nResult SA =\n{}'.format(result))
        #assert(np.max(np.abs(result_ref - result)) < 1e-9)
        #print('\nPass!')

        return result
