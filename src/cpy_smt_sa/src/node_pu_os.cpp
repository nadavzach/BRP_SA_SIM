#ifndef NODEPU_CPP
#define NODEPU_CPP

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "fifo.cpp"

#define ALU_MAX_OCP 2

using namespace std;
extern bool _node_push_back_en;
extern bool _low_prec_mult_en = false;

template <typename T>
class node_pu {
private:
    string _name;
    uint8_t _threads;
    uint8_t _alu_num;
    uint16_t _max_depth;
    T _acc = 0;
    vector<uint32_t> _acc_t;
    vector<bool> _halt;
    bool _is_util;
	float _stats_zero_ops=0;
	float _stats_1thread_mult_ops=0;
	float _stats_multi_thread_mult_ops=0;
	float _stats_alu_not_utilized=0;
	float _stats_buffer_fullness_acc=0;
	float _stats_buffer_max_fullness=0;
	int _threads_go_count=0;

public:
    vector<fifo<T>> _buf_a, _buf_b;
    node_pu *out_a, *out_b;

    node_pu (string name, uint8_t threads, uint8_t alu_num, uint16_t max_depth);
    
    T pop(bool buf, uint8_t thread, bool &is_empty);
    T get_acc() { return _acc; };
	void get_stats(float& stats_alu_not_utilized, float& stats_zero_ops,float& stats_1thread_mult_ops,float& stats_multi_thread_mult_ops,float& stats_buffer_fullness_acc,float& stats_buffer_max_fullness);
    T saturation_op(T a, T b, bool mult);




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


    // NB_SMT Project Aux Functions
    void squeeze_and_multiply(uint8_t active_threads,T alu_a_arg_arr[ALU_MAX_OCP],T alu_b_arg_arr[ALU_MAX_OCP], T& _acc);

	T pow2(int x);
	void peek(T& x,bool& is_empty, bool buf, uint8_t thread);
	bool try_pushback(uint8_t thread);
};

template <typename T>
node_pu<T>::node_pu (string name, uint8_t threads,uint8_t alu_num, uint16_t max_depth) : _name(name), _threads(threads),_alu_num(alu_num), _max_depth(max_depth), out_a(0), out_b(0) {
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
void node_pu<T>::get_stats(float& stats_alu_not_utilized,float& stats_zero_ops,float& stats_1thread_mult_ops,float& stats_multi_thread_mult_ops,float& stats_buffer_fullness_acc,float& stats_buffer_max_fullness){

	stats_zero_ops             	= _stats_zero_ops;
	stats_1thread_mult_ops     	= _stats_1thread_mult_ops;
	stats_multi_thread_mult_ops	= _stats_multi_thread_mult_ops;
	stats_buffer_fullness_acc  	= _stats_buffer_fullness_acc/_threads_go_count;
	stats_buffer_max_fullness  	= _stats_buffer_max_fullness;
	stats_alu_not_utilized     	= _stats_alu_not_utilized;

	return;

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
void node_pu<T>::peek(T& x,bool& is_empty, bool buf, uint8_t thread) {
    assert(thread < _threads);

    if (buf == 0)
       x =  _buf_a[thread].peek(is_empty);
    else
       x = _buf_b[thread].peek(is_empty);
    //if(is_empty)
    //    x = 111;
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

    bool is_a_empty;
	bool is_b_empty;
	bool th_valid_arr[_threads];
    T a[_threads];
    T b[_threads];
	int th_op_arr[_threads];// ,-1 = default, -2 pushback, -3 zero op,other - number of alu assigned 
	int alu_ocp_arr[_alu_num];//-1 = not used , else - number of mult ops used 
									
    for (uint8_t i=0; i<_alu_num; i++) {
		alu_ocp_arr[i] = -1;
	}
    for (uint8_t t=0; t<_threads; t++) {
		th_op_arr[t] = -1;
		th_valid_arr[t] =is_valid(t) && !is_halt(t) && is_ready_out(t); 

        if (th_valid_arr[t]) {

            peek(a[t],is_a_empty,0,t);
            peek(b[t],is_b_empty,1,t);
       		//assert(!is_a_empty && !is_b_empty);
            if(!is_a_empty && !is_b_empty){
                if ( (a[t] != 0 and b[t] != 0)) {
                    // check if there's a free alu
                    for(uint8_t i=0;i<_alu_num;i++){				
                        if(	alu_ocp_arr[i] == -1){
                            th_op_arr[t] = i;
                            alu_ocp_arr[i] = 1;
                            break;
                        }
                    }

                    // if there's some threads utilizing all alu - try to pushback (check buf fullness)
                    if(th_op_arr[t]<0) {

                        if(!try_pushback(t)){//try to pushback
                            //unable to pushback - 
                            //search for alu to do reduced mult
                            for(uint8_t i=0;i<_alu_num;i++){				
                                if(	alu_ocp_arr[i] < ALU_MAX_OCP){
                                    th_op_arr[t] = i;
                                    alu_ocp_arr[i]++;
                                    break;
                                }

                            }
                            assert();
                        }
                        else {// did pushback
                            th_op_arr[t] = -2;
                        }
                    }
                }
                else {
                    th_op_arr[t] = -3; //zero operation
                    //std::cout<<"a["<<(unsigned int)t<<"] = "<<(unsigned int)a[t];
                    //std::cout<<" , b["<<(unsigned int)t<<"] = "<<(unsigned int)b[t]<<std::endl;
                    _stats_zero_ops++;
                }
            }
		}
    }
    for(uint8_t t=0;t<_threads;t++){
        if((th_op_arr[t] != -2) && (th_op_arr[t] != -1) && th_valid_arr[t]){
			//assert(th_op_arr[t] == -1);
   			pop(0, t, is_a_empty);
      		pop(1, t, is_b_empty);
            if (out_a != 0){
				if(!(out_a->is_ready(0,t)))
					cout<<"ERROR!\n";
        		out_a->push(a[t], 0, t);
			}
       		if (out_b != 0){
				if(!(out_b->is_ready(1,t)))
					cout<<"ERROR!\n";

        		   out_b->push(b[t], 1, t);
			}
            _acc_t[t]++;
        }
    }
 	for (uint8_t i=0; i<_alu_num; i++) {
		T alu_a_arg_arr[ALU_MAX_OCP];
		T alu_b_arg_arr[ALU_MAX_OCP];
		uint8_t arg_arr_idx =0;
		//assert(alu_ocp_arr[i] <1);
   		for(uint8_t t=0;t<_threads;t++){
			if((th_op_arr[t] == i) && th_valid_arr[t]){
				alu_a_arg_arr[arg_arr_idx]=a[t];
				alu_b_arg_arr[arg_arr_idx]=b[t];
				arg_arr_idx++;
    		}
		}
		squeeze_and_multiply(alu_ocp_arr[i],alu_a_arg_arr,alu_b_arg_arr,_acc);
	}

	// -- statistics gather -- //

 	for (uint8_t i=0; i<_alu_num; i++) {
		if(alu_ocp_arr[i] == 1){
			_stats_1thread_mult_ops++;
		} else {
			if(alu_ocp_arr[i] > 1){
				_stats_multi_thread_mult_ops++;
			} else {
				_stats_alu_not_utilized++;
			}
		}
	}
	for (uint8_t t=0; t<_threads; t++) {
		_stats_buffer_fullness_acc = _stats_buffer_fullness_acc + (_buf_a[t].size() + _buf_b[t].size())/2;
		_threads_go_count++;
		if(_buf_a[t].size() > _stats_buffer_max_fullness)
			_stats_buffer_max_fullness = _buf_a[t].size();

		if(_buf_b[t].size() > _stats_buffer_max_fullness)
			_stats_buffer_max_fullness = _buf_b[t].size();
		//if(_buf_b[t].size() > _max_depth || _buf_a[t].size() > _max_depth )
			//std::cout<<"error - buff size is bigger than max depth, size - buf a= "<< _buf_a[t].size()<<"buf b= "<<_buf_b[t].size()<<"\n";
	}

 	//for (uint8_t t=0; t<_threads; t++) {
	//	if(th_valid_arr[t]) {
	//		cout<<" threads op arr["<<(unsigned int)t<<"]= "<<th_op_arr[t]<<"\n";
	//		cout<<" a["<<(unsigned int)t<<"]= "<<(T)a[t]<<"\n";
	//		cout<<" b["<<(unsigned int)t<<"]= "<<(T)b[t]<<"\n";
	//	} else
	//	{
	//		cout<<" at least one of the node args is empty"<<"\n";
	//	}
	//}
}


template <typename T>
bool node_pu<T>::try_pushback(uint8_t thread) {
	if (!_node_push_back_en || _buf_a[thread].size() >= _max_depth-1 || _buf_b[thread].size()  >= _max_depth-1)
		return false;
	return true;	
}

template <typename T>
void node_pu<T>::squeeze_and_multiply(uint8_t active_threads,T alu_a_arg_arr[ALU_MAX_OCP],T alu_b_arg_arr[ALU_MAX_OCP],T& _acc){
    int bits = sizeof(T)*8;
    int half_bits = bits/2;
    T max_half_T_size = pow2(bits/2);
    T max_quarter_T_size = max_half_T_size/2;
    if(active_threads == 2){
        for (uint8_t t=0; t < active_threads; t++) {
            unsigned int a = (unsigned int)alu_a_arg_arr[t];
            unsigned int b = (unsigned int)alu_b_arg_arr[t];
            unsigned int a_msb = a / max_half_T_size;
            unsigned int a_lsb = (a <<half_bits)>>half_bits;
            unsigned int b_msb = b / max_half_T_size;
            unsigned int b_lsb = (b<<half_bits)>>half_bits;
            if(a_msb*a_lsb*b_msb != 0){
                if(a_lsb >= max_quarter_T_size) 
                    a_msb += 1; // rounding a's MSB
                _acc = saturation_op(_acc, (saturation_op(a_msb, b, true) << (bits/2)), false);
            }
            else if(a_msb == 0){
                _acc = saturation_op(_acc, saturation_op(a_lsb, b, true), false);
                //_acc += a_lsb * b;
            }
            else if(a_lsb == 0){
                _acc = saturation_op(_acc, (saturation_op(a_msb, b, true) << (bits/2)), false);
                //_acc += (a_msb * b) << (bits/2);
            }
            else{
                _acc = saturation_op(_acc, saturation_op(a, b_lsb, true), false);
                //_acc += a * b_lsb;
            }
        }
    }
    else if(active_threads == 1){
        T a,b;
        for (uint8_t t=0; t < _threads; t++){
            if(alu_a_arg_arr[t] != 0){
                a = alu_a_arg_arr[t];
                b = alu_b_arg_arr[t];
                break;
            }
        }
        _acc = saturation_op(_acc, saturation_op(a, b, true), false);
        //_acc = a * b;
    }
}

template <typename T>
T node_pu<T>::pow2(int x){
    T retval = 1;
    for(int i = 0 ; i<x ; i++)
        retval *= 2;
    return retval;
}
//T low_prec_mult(T a[_threads],T b[_threads]) {
//	for(uint8_t t=0; t<_threads; t++){
//	}
//
//	
//}

//T round_64bit(T x){
//	if(a[t] > 2**32-1)
//		return x>>32;
//	else
//		return x<<32;
//	
//}

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
template <typename T>
T node_pu<T>::saturation_op(T a, T b, bool mult)
{
    int bits = sizeof(T)*8;
    int64_t max_sat = pow2(bits)-1;
    int64_t min_sat = 0;
    int64_t temp_res;
    if(mult){
        temp_res = a*b;
    }
    else
        temp_res = a+b;
    if(temp_res < a)
        return (T)max_sat;
    return (T)temp_res;
}
#endif
