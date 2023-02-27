#include <iostream>
#include <vector>
#include <cmath>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xrandom.hpp"
#include "grid_os.cpp"

using namespace std;

struct tile_idx {
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;
};
int total_ops = 0;

template <typename T>
class smt_sa_os {
private:
    uint16_t _dim;
    uint8_t _threads;
    uint16_t _max_depth;
    xt::xarray<T> _a;
    xt::xarray<T> _b;

    void _subtile_dict(vector<uint16_t> &subtile_start, vector<uint16_t> &subtile_end);
    void _subtile_range(uint8_t thread, uint16_t &thread_tile_start, uint16_t &thread_tile_end);

public:
    grid<T> sa_grid;
    uint64_t cycles;

    smt_sa_os (uint16_t dim, uint8_t threads, uint16_t max_depth=4096);

    void set_inputs(xt::xarray<T> a, xt::xarray<T> b);
    void get_tile(vector<xt::xarray<T>> &tile_a, vector<xt::xarray<T>> &tile_b, tile_idx t_idx);
    xt::xarray<T> go();
    xt::xarray<T> go(vector<tile_idx> &tile_vec);
};

template <typename T>
smt_sa_os<T>::smt_sa_os (uint16_t dim, uint8_t threads, uint16_t max_depth) : _dim(dim), _threads(threads), _max_depth(max_depth), sa_grid(dim, threads, max_depth), cycles(0) {}

template <typename T>
void smt_sa_os<T>::set_inputs(xt::xarray<T> a, xt::xarray<T> b) {
    assert(a.shape().size() == 3);
    assert(b.shape().size() == 2);
    assert(a.shape()[2] == b.shape()[0]);

    _a = a;
    _b = b; 
}

template <typename T>
void smt_sa_os<T>::get_tile(vector<xt::xarray<T>> &tile_a, vector<xt::xarray<T>> &tile_b, tile_idx t_idx) {
    uint16_t a_tile_H = _dim;
    uint16_t b_tile_W = _dim;

    vector<uint16_t> subtile_start;
    vector<uint16_t> subtile_end;
    _subtile_dict(subtile_start, subtile_end);

    for (uint8_t t=0; t<_threads; t++) {
        xt::xarray<T> tile = xt::view(_a, t_idx.d1, xt::range(t_idx.d2 * a_tile_H, (t_idx.d2+1)*a_tile_H), xt::range(subtile_start[t], subtile_end[t]));
        tile_a.push_back(tile);

        if (_dim > tile_a[t].shape()[0]) {
            tile_a[t] = xt::pad(tile_a[t], {{0, _dim - tile_a[t].shape()[0]}, {0, 0}});
        }
    }

    for (uint8_t t=0; t<_threads; t++) {
        xt::xarray<T> tile = xt::view(_b, xt::range(subtile_start[t], subtile_end[t]), xt::range(t_idx.d3*b_tile_W, (t_idx.d3+1)*b_tile_W));
        tile_b.push_back(tile);

        if (_dim > tile_b[t].shape()[1]) {
            tile_b[t] = xt::pad(tile_b[t], {{0, 0}, {0, _dim - tile_b[t].shape()[1]}});
        }
    }
}

template <typename T>
void smt_sa_os<T>::_subtile_dict(vector<uint16_t> &subtile_start, vector<uint16_t> &subtile_end) {
    for (uint8_t t=0; t<_threads; t++) {
        uint16_t start, end;
        _subtile_range(t, start, end);
        subtile_start.push_back(start);
        subtile_end.push_back(end);
    }
}

template <typename T>
void smt_sa_os<T>::_subtile_range(uint8_t thread, uint16_t &thread_tile_start, uint16_t &thread_tile_end) {
    uint16_t a_tile_W = _a.shape()[2];
    uint16_t b_tile_H = _b.shape()[0];
    assert(a_tile_W == b_tile_H);

    uint16_t subtile_size = floor(float(b_tile_H) / _threads);

    if (thread < _threads - 1) {
        thread_tile_start = thread * subtile_size;
        thread_tile_end = (thread+1) * subtile_size;
    }
    else {
        thread_tile_start = thread * subtile_size;
        thread_tile_end = b_tile_H;
    }
}

template <typename T>
xt::xarray<T> smt_sa_os<T>::go(vector<tile_idx> &tile_vec) {
    assert(tile_vec.size() > 0);
    uint16_t a_tiles = ceil(float(_a.shape()[1]) / _dim);
    uint16_t b_tiles = ceil(float(_b.shape()[1]) / _dim);

    // Assuming tile_vec is ordered (batch, height, width), i.e., rows->columns->depth!
    xt::xarray<uint32_t> array_ctrl = xt::ones<uint32_t>({_dim, _dim}) * ((tile_vec[0].d1 * a_tiles * b_tiles) + (tile_vec[0].d2 * b_tiles) + (tile_vec[0].d3));

    uint32_t global_tile_idx = 1;
    vector<xt::xarray<T>> tile_a, tile_b;
    get_tile(tile_a, tile_b, tile_vec[0]);
    for (uint8_t t=0; t<_threads; t++)
        sa_grid.push(tile_a[t], tile_b[t], t, true);

    xt::xarray<T> result = xt::zeros<T>({_a.shape()[0], _a.shape()[1], _b.shape()[1]});

    vector<uint16_t> subtile_start, subtile_end;
    _subtile_dict(subtile_start, subtile_end);

    uint32_t computed = 0;
    uint32_t while_end = tile_vec.size() * _dim * _dim;

    while (computed < while_end) {
        sa_grid.cycle();
        cycles++;

        for (uint16_t i=0; i<_dim; i++) {
            for (uint16_t j=0; j<_dim; j++) {
                uint8_t halt_count = 0;

                for (uint8_t t=0; t<_threads; t++) {
                    if (sa_grid.nodes[i][j].is_halt(t))
                        halt_count++;

                    uint32_t acc_t = sa_grid.nodes[i][j].get_acc_t(t);

                    assert(subtile_end[t] - subtile_start[t] >= 0);

                    if ((acc_t == uint32_t(subtile_end[t] - subtile_start[t])) && !sa_grid.nodes[i][j].is_halt(t))
                        sa_grid.nodes[i][j].halt(t);
                }

                if (halt_count == _threads) {
                    uint32_t batch = floor(float(array_ctrl(i, j)) / (a_tiles * b_tiles));
                    uint32_t i_result = int(i + int((array_ctrl(i, j) % (a_tiles * b_tiles)) / b_tiles) * _dim);
                    uint32_t j_result = int(j + ((array_ctrl(i, j) % (a_tiles * b_tiles)) % b_tiles) * _dim);

                    if (i_result < result.shape()[1] && j_result < result.shape()[2])
                        result(batch, i_result, j_result) = sa_grid.nodes[i][j].get_acc();
                    
                    array_ctrl(i, j)++;
                    sa_grid.nodes[i][j].reset_acc();
                    sa_grid.nodes[i][j].reset_acc_t();
                    computed++;

                    sa_grid.nodes[i][j].release();
                }
            }
        }

        if (sa_grid.mem_a[0]._buf[0].size() < 128) {
            if (global_tile_idx < tile_vec.size()) {
                tile_a.clear();
                tile_b.clear();

                get_tile(tile_a, tile_b, tile_vec[global_tile_idx]);

                for (uint8_t t=0; t<_threads; t++)
                    sa_grid.push(tile_a[t], tile_b[t], t);

                global_tile_idx++;
            }
        }
    }
	std::cout<<"Total Cycles = "<<cycles<<std::endl;
	std::cout<<"Total Ops = "<<total_ops<<std::endl;
    return result;
}

template <typename T>
xt::xarray<T> smt_sa_os<T>::go() {
    uint16_t batch = _a.shape()[0];
    uint16_t a_tiles = ceil(float(_a.shape()[1]) / _dim);
    uint16_t b_tiles = ceil(float(_b.shape()[1]) / _dim);

    xt::xarray<T> result = xt::zeros<T>({_a.shape()[0], _a.shape()[1], _b.shape()[1]});
    vector<tile_idx> tile_vec;

    for (uint16_t b=0; b<batch; b++) {
        for (uint16_t i=0; i<a_tiles; i++) {
            for (uint16_t j=0; j<b_tiles; j++) {
                tile_idx t_idx;
                t_idx.d1 = b;
                t_idx.d2 = i;
                t_idx.d3 = j;
                tile_vec.push_back(t_idx);
            }
        }
    }

    result += go(tile_vec);

    return result;
}
