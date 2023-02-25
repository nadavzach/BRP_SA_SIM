#ifndef NODEPUWS_CPP
#define NODEPUWS_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "node.cpp"
#include "fifo.cpp"

using namespace std;

template <typename T>
class node_pu_ws: public node<T> {
private:
    T _weight = 0;

public:
    node_pu_ws (string name, uint8_t threads, uint16_t max_depth);

    void set_weight(T w) { _weight = w; };

    void go();
};

template <typename T>
node_pu_ws<T>::node_pu_ws (string name, uint8_t threads, uint16_t max_depth) : node<T>(name, threads, max_depth) {}

template <typename T>
void node_pu_ws<T>::go() {
    this->_is_util = false;

    for (uint8_t t=0; t<this->_threads; t++) {
        if (this->is_valid(t) && this->is_ready_out(t)) {
            bool is_a_empty, is_b_empty;
            record<T> x = this->pop(0, t, is_a_empty);
            record<T> psum = this->pop(1, t, is_b_empty);
            
            assert(!is_a_empty && !is_b_empty);

            if (x.data != 0 and psum.data != 0) {
                if (this->out_a != 0)
                    this->out_a->push(x, 0, t);
                    
                if (this->out_b != 0) {
                    record<T> rec;
                    rec.valid = psum.valid;
                    rec.data = psum.data + x.data * _weight;
                    this->out_b->push(rec, 1, t);
                }
            }
        }
    }
}

#endif