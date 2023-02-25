#include <iostream>
#include <vector>
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "node_pu_ws.cpp"
#include "node_mem.cpp"
#include "node_acc.cpp"

using namespace std;

template <typename T>
class grid_ws {
private:
    uint16_t _dim;
    uint8_t _threads;
    uint16_t _max_depth;
public:
    vector<node_mem<T>> mem_a;
    vector<node_mem<T>> mem_b;
    vector<vector<node_pu_ws<T>>> nodes;
    vector<node_acc<T>> acc;

    grid_ws (uint16_t dim, uint8_t threads, uint16_t max_depth=0);
    void push(xt::xarray<T> &a, xt::xarray<T> &b, uint8_t thread, bool pad=false);
    void cycle();
    void get_util_rate();
};

template <typename T>
grid_ws<T>::grid_ws (uint16_t dim, uint8_t threads, uint16_t max_depth) : _dim(dim), _threads(threads), _max_depth(max_depth) {
    mem_a.reserve(dim);
    mem_b.reserve(dim);
    nodes.reserve(dim);
    acc.reserve(dim);

    for (uint8_t i=0; i<dim; i++) {
        mem_a.push_back(node_mem<T>("mem_a_" + to_string(i), threads));
        mem_b.push_back(node_mem<T>("mem_b_" + to_string(i), threads));
        
        acc.push_back(node_acc<T>("acc_" + to_string(i), threads));
        
        nodes.push_back(vector<node_pu_ws<T>>());
        nodes[i].reserve(dim);

        for (uint8_t j=0; j<dim; j++) {
            nodes[i].push_back(node_pu_ws<T>("node_" + to_string(i) + "_" + to_string(j), threads, max_depth));
        }
    }

    // Connect PUs
    for (uint16_t i=0; i<dim; i++) {
        for (uint16_t j=0; j<dim; j++) {
            if (j+1 < dim)
                nodes[i][j].out[0] = &nodes[i][j+1];

            if (i+1 < dim)
                nodes[i][j].out[1] = &nodes[i+1][j];
            else
                nodes[i][j].out[1] = &acc[i];
        }
    }

    // Connect memory units
    for (uint16_t i=0; i<dim; i++) {
        mem_a[i].out[0] = &nodes[i][0];
        mem_a[i].out_buf_idx = 0;

        mem_b[i].out[0] = &nodes[0][i];
        mem_b[i].out_buf_idx = 1;
    }
}

template <typename T>
void grid_ws<T>::push(xt::xarray<T> &a, xt::xarray<T> &b, uint8_t thread, bool pad) {
    uint32_t a_H = a.shape()[0];
    uint32_t a_W = a.shape()[1];
    uint32_t b_H = b.shape()[0];
    uint32_t b_W = b.shape()[1];
    assert(a_W == b_H);

    for (uint32_t i=0; i<a_H; i++) {
        if (pad) {
            for (uint32_t k=0; k<i; k++) {
                record<T> me;
                me.valid = false;
                me.data = 0;
                mem_a[i]._buf[0][thread]._phy_fifo.push(me);
                mem_a[i]._buf[0][thread]._arch_fifo.push(me);
            }
        }

        for (uint32_t j=0; j<a_W; j++) {
            record<T> me;
            me.valid = true;
            me.data = a(i, j);
            mem_a[i]._buf[0][thread]._phy_fifo.push(me);
            mem_a[i]._buf[0][thread]._arch_fifo.push(me);
        }
    }

    for (uint32_t j=0; j<b_W; j++) {
        if (pad) {
            for (uint32_t k=0; k<j; k++) {
                record<T> me;
                me.valid = false;
                me.data = 0;
                mem_b[j]._buf[0][thread]._phy_fifo.push(me);
                mem_b[j]._buf[0][thread]._arch_fifo.push(me);
            }
        }

        for (uint32_t i=0; i<b_H; i++) {
            record<T> me;
            me.valid = true;
            me.data = b(i, j);
            mem_b[j]._buf[0][thread]._phy_fifo.push(me);
            mem_b[j]._buf[0][thread]._arch_fifo.push(me);
        }
    }
}

template <typename T>
void grid_ws<T>::cycle() {
    for (uint16_t i=0; i<_dim; i++) {
        for (uint16_t j=0; j<_dim; j++) {
            nodes[i][j].go();

            for (uint8_t t=0; t<_threads; t++) {
                nodes[i][j]._buf[0][t].cycle();
                nodes[i][j]._buf[1][t].cycle();
            }
        }
    }

    for (uint16_t i=0; i<_dim; i++) {
        mem_a[i].go();
        mem_b[i].go();
        acc[i].go();

        mem_a[i].cycle();
        mem_b[i].cycle();
        acc[i].cycle();
    }
}
