#ifndef NODEPUOS_CPP
#define NODEPUOS_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "node.cpp"
#include "fifo.cpp"

using namespace std;

template <typename T>
class node_pu_os: public node<T> {
private:
    T _acc = 0;
    vector<uint32_t> _acc_t;
    vector<bool> _halt;

public:
    node_pu_os (string name, uint8_t threads, uint16_t max_depth);
    
    T get_acc() { return _acc; };
    uint32_t get_acc_t(uint8_t thread) { return _acc_t[thread]; }; 

    void go();

    void reset_acc() { _acc = 0; };
    void reset_acc_t();
    void reset_acc_t(uint8_t thread) { _acc_t[thread] = 0; };
    void halt(uint8_t thread) { _halt[thread] = true; };
    void release();
    void release(uint8_t thread) { _acc_t[thread] = false; };
    bool is_halt(uint8_t thread) { return _halt[thread]; };
};

template <typename T>
node_pu_os<T>::node_pu_os (string name, uint8_t threads, uint16_t max_depth) : node<T>(name, 2, threads, max_depth) {
    _acc_t.reserve(threads);
    _halt.reserve(threads);

    for (uint8_t i=0; i<threads; i++) {
        _acc_t.push_back(0);
        _halt.push_back(false);
    }
}

template <typename T>
void node_pu_os<T>::go() {
    this->_is_util = false;

    for (uint8_t t=0; t<this->_threads; t++) {
        if (this->is_valid(t) && !is_halt(t) && this->is_ready_out(t)) {
            bool is_a_empty, is_b_empty;
            record<T> a = this->pop(0, t, is_a_empty);
            record<T> b = this->pop(1, t, is_b_empty);
            
            assert(!is_a_empty && !is_b_empty);

            _acc_t[t]++;

            if (this->out[0] != 0)
                this->out[0]->push(a, 0, t);
            if (this->out[1] != 0)
                this->out[1]->push(b, 1, t);

            if (a.data != 0 and b.data != 0) {
                _acc += a.data * b.data;
                this->_is_util = true;
                break;
            }
        }
    }
}

template <typename T>
void node_pu_os<T>::reset_acc_t() {
    for (uint8_t t=0; t<this->_threads; t++)
        this->_acc_t[t] = 0;
}

template <typename T>
void node_pu_os<T>::release() {
    for (uint8_t t=0; t<this->_threads; t++)
        this->_halt[t] = false;
}

#endif