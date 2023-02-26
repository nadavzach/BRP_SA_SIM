#ifndef NODEMEM_CPP
#define NODEMEM_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "fifo.cpp"
#include "node_pu_os.cpp"

using namespace std;

template <typename T>
struct mem_entry {
    bool valid;
    T data;
};

template <typename T>
class node_mem {
private:
    string _name;
    uint8_t _threads;
    
public:
    vector<fifo<mem_entry<T>>> _buf;
    vector<uint16_t> _skip_counter;
    node_pu<T> *out;
    bool out_buf_idx;

    node_mem (string name, uint8_t threads);
    
    mem_entry<T> pop(uint8_t thread, bool &is_empty) { assert(thread < _threads); return _buf[thread].pop(is_empty); };
    void push(mem_entry<T> x, bool buf, uint8_t thread) { assert(thread < _threads); _buf[thread].push(x); };
    void go();
};

template <typename T>
node_mem<T>::node_mem (string name, uint8_t threads) : _name(name), _threads(threads) {
    _buf.reserve(threads);

    for (uint8_t i=0; i<threads; i++)
        _buf.push_back(fifo<mem_entry<T>>("fifo_a_" + name));
}

template <typename T>
void node_mem<T>::go() {
    for (uint8_t t=0; t<_threads; t++) {
        if (_buf[t].size() > 0 && out->is_ready(out_buf_idx,t)) {
            bool is_empty;
            mem_entry<T> me = pop(t, is_empty);

            if (me.valid)
                out->push(me.data, out_buf_idx, t);
        }
    }
}

#endif
