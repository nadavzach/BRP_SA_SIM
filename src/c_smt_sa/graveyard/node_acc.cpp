#ifndef NODEACC_CPP
#define NODEACC_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "fifo.cpp"
#include "node.cpp"

using namespace std;


template <typename T>
class node_acc: public node<T> {
private:
    
public:
    bool out_buf_idx;

    node_acc (string name, uint8_t threads);
    
    void go();
};

template <typename T>
node_acc<T>::node_acc (string name, uint8_t threads) : node<T>(name, 1, threads) {}

template <typename T>
void node_acc<T>::go() {
    for (uint8_t t=0; t<this->_threads; t++) {
        if (this->_buf[0][t].size() > 0) {
            bool is_empty;
            record<T> rec = this->pop(0, t, is_empty);

            cout << this->_name << ": " << rec.data << endl;
        }
    }
}

#endif