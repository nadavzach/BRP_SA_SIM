#ifndef FIFO_CPP
#define FIFO_CPP

#include <iostream>
#include <string>
#include <queue>
#include <cassert>

using namespace std;

template <typename T>
class fifo {
private:
    string _name;

    T _last_pushed;
    bool _pushed;
    bool _popped;
public:
    queue<T> _arch_fifo;
    queue<T> _phy_fifo;
    
    fifo (string name);
    T peek(bool& is_empty);
    void push(T x);
    T pop(bool& is_empty);
    uint32_t size();
    void cycle();
};

template <typename T>
fifo<T>::fifo (string name) : _name(name), _pushed(false), _popped(false) {}

template <typename T>
T fifo<T>::peek(bool& is_empty) {
    if (_arch_fifo.size() == 0) {
        is_empty = true;
        // TODO: should probably return something
    }
    else {
        is_empty = false;
        return _arch_fifo.front();
    }
}

template <typename T>
void fifo<T>::push(T x) {
    assert(_pushed == false);

    _phy_fifo.push(x);
    _pushed = true;
    _last_pushed = x;
}

template <typename T>
T fifo<T>::pop(bool& is_empty) {
    assert(_popped == false);

    if (_phy_fifo.size() > 0)
        _phy_fifo.pop();

    _popped = true;
    return peek(is_empty);
}

template <typename T>
uint32_t fifo<T>::size() {
    return _arch_fifo.size();
}

template <typename T>
void fifo<T>::cycle() {
    if (_pushed) {
        _arch_fifo.push(_last_pushed);
        _pushed = false;
    }

    if (_popped) {
        _arch_fifo.pop();
        _popped = false;
    }

    assert(_arch_fifo.size() == _phy_fifo.size());
}

#endif