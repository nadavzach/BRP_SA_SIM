#include <iostream>
#include <vector>
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "node_pu_os.cpp"
#include "node_mem.cpp"

using namespace std;
extern bool _run_parallel;
template <typename T>
class grid {
private:
    uint16_t _dim;
    uint8_t _threads;
    uint8_t _alu_num;
    uint16_t _max_depth;
public:
    vector<node_mem<T>> mem_a;
    vector<node_mem<T>> mem_b;
    vector<vector<node_pu<T>>> nodes;

    grid (uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth=4096);
    void push(xt::xarray<T> &a, xt::xarray<T> &b, uint8_t thread, bool pad=false);
    void cycle();
    void get_util_rate();
};

template <typename T>
grid<T>::grid (uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth) : _dim(dim), _threads(threads),_alu_num(alu_num), _max_depth(max_depth) {
    mem_a.reserve(dim);
    mem_b.reserve(dim);
    nodes.reserve(dim);

    for (uint8_t i=0; i<dim; i++) {
        mem_a.push_back(node_mem<T>("mem_a_" + to_string(i), threads));
        mem_b.push_back(node_mem<T>("mem_b_" + to_string(i), threads));
        
        nodes.push_back(vector<node_pu<T>>());
        nodes[i].reserve(dim);

        for (uint8_t j=0; j<dim; j++) {
            nodes[i].push_back(node_pu<T>("node_" + to_string(i) + "_" + to_string(j), threads,_alu_num, max_depth));
        }
    }

    // Connect PUs
    for (uint16_t i=0; i<dim; i++) {
        for (uint16_t j=0; j<dim; j++) {
            if (j+1 < dim)
                nodes[i][j].out_a = &nodes[i][j+1];
            if (i+1 < dim)
                nodes[i][j].out_b = &nodes[i+1][j];
        }
    }

    // Connect memory units
    for (uint16_t i=0; i<dim; i++) {
        mem_a[i].out = &nodes[i][0];
        mem_a[i].out_buf_idx = 0;

        mem_b[i].out = &nodes[0][i];
        mem_b[i].out_buf_idx = 1;
    }
}

template <typename T>
void grid<T>::push(xt::xarray<T> &a, xt::xarray<T> &b, uint8_t thread, bool pad) {
    
    uint32_t a_H = a.shape()[0];
    uint32_t a_W = a.shape()[1];
    uint32_t b_H = b.shape()[0];
    uint32_t b_W = b.shape()[1];
    assert(a_W == b_H);

    for (uint32_t i=0; i<a_H; i++) {
        if (pad) {
            for (uint32_t k=0; k<i; k++) {
                mem_entry<T> me;
                me.valid = false;
                me.data = 0;
                mem_a[i]._buf[thread]._phy_fifo.push(me);
                mem_a[i]._buf[thread]._arch_fifo.push(me);
            }
        }

        for (uint32_t j=0; j<a_W; j++) {
            mem_entry<T> me;
            me.valid = true;
            me.data = a(i, j);
            mem_a[i]._buf[thread]._phy_fifo.push(me);
            mem_a[i]._buf[thread]._arch_fifo.push(me);
        }
    }

    for (uint32_t j=0; j<b_W; j++) {
        if (pad) {
            for (uint32_t k=0; k<j; k++) {
                mem_entry<T> me;
                me.valid = false;
                me.data = 0;
                mem_b[j]._buf[thread]._phy_fifo.push(me);
                mem_b[j]._buf[thread]._arch_fifo.push(me);
            }
        }

        for (uint32_t i=0; i<b_H; i++) {
            mem_entry<T> me;
            me.valid = true;
            me.data = b(i, j);
            mem_b[j]._buf[thread]._phy_fifo.push(me);
            mem_b[j]._buf[thread]._arch_fifo.push(me);
        }
    }
}

template <typename T>
void grid<T>::cycle() {
    if(_run_parallel){
        //#pragma omp parallel for
        for (uint16_t i=0; i<_dim; i++) {
            //#pragma omp parallel for
            for (uint16_t j=0; j<_dim; j++) {
                nodes[i][j].go();

                for (uint8_t t=0; t<_threads; t++) {
                    nodes[i][j]._buf_a[t].cycle();
                    nodes[i][j]._buf_b[t].cycle();
                }
            }
        }
    }
    else{
        for (uint16_t i=0; i<_dim; i++) {
            for (uint16_t j=0; j<_dim; j++) {
                nodes[i][j].go();

                for (uint8_t t=0; t<_threads; t++) {
                    nodes[i][j]._buf_a[t].cycle();
                    nodes[i][j]._buf_b[t].cycle();
                }
            }
        }
    }

    for (uint16_t i=0; i<_dim; i++) {
        mem_a[i].go();
        mem_b[i].go();

        for (uint8_t t=0; t<_threads; t++) {
            mem_a[i]._buf[t].cycle();
            mem_b[i]._buf[t].cycle();
        }
    }
}
