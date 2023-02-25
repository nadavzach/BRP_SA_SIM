#ifndef NODEPU_CPP
#define NODEPU_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "fifo.cpp"

using namespace std;

template <typename T>
class node_pu {
private:
    string _name;
    uint8_t _threads;
    uint16_t _max_depth;
    T _acc = 0;
    vector<uint32_t> _acc_t;
    vector<bool> _halt;
    bool _is_util;

public:
    vector<fifo<T>> _buf_a, _buf_b;
    node_pu *out_a, *out_b;

    node_pu (string name, uint8_t threads, uint16_t max_depth);
    
    T pop(bool buf, uint8_t thread, bool &is_empty);
    T get_acc() { return _acc; };

    uint32_t get_acc_t(uint8_t thread) { return _acc_t[thread]; }; 

    void push(T x, bool buf, uint8_t thread);
    void reset_acc() { _acc = 0; };
    void reset_acc_t();
    void reset_acc_t(uint8_t thread) { _acc_t[thread] = 0; };
    void halt(uint8_t thread) { _halt[thread] = true; };
    void release();
    void release(uint8_t thread) { _acc_t[thread] = false; };
    void cycle();
    void go();

    bool is_valid(uint8_t thread);
    bool is_ready(bool buf, uint8_t thread);
    bool is_ready_out(uint8_t thread);
    bool is_halt(uint8_t thread) { return _halt[thread]; };
    bool is_util() { return _is_util; };
};

template <typename T>
node_pu<T>::node_pu (string name, uint8_t threads, uint16_t max_depth) : _name(name), _threads(threads), _max_depth(max_depth), out_a(0), out_b(0) {
    _buf_a.reserve(threads);
    _buf_b.reserve(threads);
    _acc_t.reserve(threads);
    _halt.reserve(threads);

    for (uint8_t i=0; i<threads; i++) {
        _buf_a.push_back(fifo<T>("fifo_a_" + name));
        _buf_b.push_back(fifo<T>("fifo_b_" + name));
        _acc_t.push_back(0);
        _halt.push_back(false);
    }
}

template <typename T>
void node_pu<T>::push(T x, bool buf, uint8_t thread) {
    assert(thread < _threads);

    if (buf == 0)
        _buf_a[thread].push(x);
    else
        _buf_b[thread].push(x);
}

template <typename T>
T node_pu<T>::pop(bool buf, uint8_t thread, bool &is_empty) {
    assert(thread < _threads);

    if (buf == 0)
        return _buf_a[thread].pop(is_empty);
    else
        return _buf_b[thread].pop(is_empty);
}

template <typename T>
bool node_pu<T>::is_valid(uint8_t thread) {
    if (_buf_a[thread].size() == 0 || _buf_b[thread].size() == 0)
        return false;
    else
        return true;
}

template <typename T>
bool node_pu<T>::is_ready(bool buf, uint8_t thread) {
    if ((buf == 0 && _buf_a[thread].size() < _max_depth) || (buf == 1 && _buf_b[thread].size() < _max_depth))
        return true;
    else
        return false;
}

template <typename T>
bool node_pu<T>::is_ready_out(uint8_t thread) {
    bool a_out_ready, b_out_ready;

    if (out_a == 0)
        a_out_ready = true;
    else
        a_out_ready = out_a->is_ready(0, thread);

    if (out_b == 0)
        b_out_ready = true;
    else
        b_out_ready = out_b->is_ready(1, thread);

    return a_out_ready && b_out_ready;
}

template <typename T>
void node_pu<T>::go() {
    _is_util = false;

    for (uint8_t t=0; t<_threads; t++) {
        if (is_valid(t) && !is_halt(t) && is_ready_out(t)) {
            bool is_a_empty, is_b_empty;
            T a = pop(0, t, is_a_empty);
            T b = pop(1, t, is_b_empty);
            
            assert(!is_a_empty && !is_b_empty);

            _acc_t[t]++;

            if (out_a != 0)
                out_a->push(a, 0, t);
            if (out_b != 0)
                out_b->push(b, 1, t);

            if (a != 0 and b != 0) {
                _acc += a * b;
                _is_util = true;
                break;
            }
        }
    }
}

template <typename T>
void node_pu<T>::reset_acc_t() {
    for (uint8_t t=0; t<_threads; t++)
        _acc_t[t] = 0;
}

template <typename T>
void node_pu<T>::release() {
    for (uint8_t t=0; t<_threads; t++)
        _halt[t] = false;
}

template <typename T>
void node_pu<T>::cycle() {
    for (uint8_t t=0; t<_threads; t++) {
        _buf_a[t].cycle();
        _buf_b[t].cycle();
    }
}

#endif