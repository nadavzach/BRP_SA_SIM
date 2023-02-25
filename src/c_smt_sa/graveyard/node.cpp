#ifndef NODE_CPP
#define NODE_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "fifo.cpp"

using namespace std;


template <typename T>
struct record {
    bool valid;
    T data;
};


template <typename T>
class node {
protected:
    string _name;
    uint8_t _buffers;
    uint8_t _threads;
    uint16_t _max_depth;
    bool _is_util;

public:
    vector<vector<fifo<record<T>>>> _buf;     // [buffers][threads]; TODO: make buffers private
    vector<node*> out;

    node (string name, uint8_t buffers=1, uint8_t threads=1, uint16_t max_depth=0);
    
    virtual void go() =0;

    record<T> pop(uint8_t buf, uint8_t thread, bool &is_empty);
    void push(record<T> x, uint8_t buf, uint8_t thread);
    void cycle();

    bool is_valid(uint8_t thread);
    bool is_ready(uint8_t buf, uint8_t thread);
    bool is_ready_out(uint8_t thread);
    bool is_util() { return _is_util; };
};

template <typename T>
node<T>::node (string name, uint8_t buffers, uint8_t threads, uint16_t max_depth) : _name(name), _buffers(buffers), _threads(threads), _max_depth(max_depth) {
    _buf.reserve(buffers);
    out.reserve(buffers);

    for (uint8_t b=0; b<buffers; b++) {
        out.push_back(0);

        _buf.push_back(vector<fifo<record<T>>>());
        _buf[b].reserve(threads);

        for (uint8_t i=0; i<threads; i++) {
            _buf[b].push_back(fifo<record<T>>("fifo_" + to_string(b) + "_" + name));
        }
    }
}

template <typename T>
void node<T>::push(record<T> x, uint8_t buf, uint8_t thread) {
    assert(thread < _threads);

    _buf[buf][thread].push(x);
}

template <typename T>
record<T> node<T>::pop(uint8_t buf, uint8_t thread, bool &is_empty) {
    assert(thread < _threads);

    return _buf[buf][thread].pop(is_empty);
}

template <typename T>
bool node<T>::is_valid(uint8_t thread) {
    for (uint8_t b=0; b<_buffers; b++) {
        if (_buf[b][thread].size() == 0)
            return false;
    }

    return true;
}

template <typename T>
bool node<T>::is_ready(uint8_t buf, uint8_t thread) {
    if (_buf[buf][thread].size() < _max_depth || _max_depth == 0)
        return true;
    else
        return false;
}

template <typename T>
bool node<T>::is_ready_out(uint8_t thread) {
    bool a_out_ready, b_out_ready;

    if (out[0] == 0)
        a_out_ready = true;
    else
        a_out_ready = out[0]->is_ready(0, thread);

    if (out[1] == 0)
        b_out_ready = true;
    else
        b_out_ready = out[1]->is_ready(1, thread);

    return a_out_ready && b_out_ready;
}

template <typename T>
void node<T>::cycle() {
    for(uint8_t b=0; b<_buffers; b++) {
        for (uint8_t t=0; t<_threads; t++)
            _buf[b][t].cycle();
    }
}

#endif