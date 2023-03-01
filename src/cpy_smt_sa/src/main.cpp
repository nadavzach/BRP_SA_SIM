#include "pybind11/pybind11.h"

#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include "smt_sa_os.cpp"

#include <iostream>

namespace py = pybind11;
bool _node_push_back_en = false;
bool _node_low_prec_mult_en = false;
bool _run_parallel = false;
uint64_t max_number;
uint64_t max_number_half_bits;
int signed_max_number; 
int signed_min_number; 


int pow2(int x){
    int retval = 1;
    for(int i = 0 ; i<x ; i++)
        retval *= 2;
    return retval;
}

inline std::tuple<xt::pyarray<uint8_t>,float , float ,float, float ,float ,float ,float > run_uint8(
	uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth, xt::pyarray<uint8_t> &a, xt::pyarray<uint8_t> &b,
	bool push_back_en, bool low_prec_mult_en, bool run_parallel)
{
	int bits = sizeof(uint8_t)*8;
	max_number = pow2(bits)-1;
	max_number_half_bits = pow2(bits/2);
	signed_max_number = pow2(bits)/2-1;
	signed_min_number = -pow2(bits)/2;
	cout<<"signed max and min: "<<signed_max_number<< " " <<signed_min_number;
	
	stats_str stats;
	_node_push_back_en = push_back_en;
	_node_low_prec_mult_en = low_prec_mult_en;
	_run_parallel = run_parallel;
    smt_sa_os<uint8_t> sa(dim, threads,alu_num, max_depth);
    sa.set_inputs(a, b);
	auto start = high_resolution_clock::now();
	xt::pyarray<uint8_t> res = sa.go(stats);
    auto end = high_resolution_clock::now();
	cout<<"$$$$ Total test runtime   :   "<<duration_cast<microseconds>(end-start).count()<<"[us]"<<endl;

	float stats_zero_ops             	= stats.stats_zero_ops;
	float stats_1thread_mult_ops     	= stats.stats_1thread_mult_ops;
	float stats_multi_thread_mult_ops	= stats.stats_multi_thread_mult_ops;
	float stats_buffer_fullness_acc  	= stats.stats_buffer_fullness_acc;
	float stats_buffer_max_fullness  	= stats.stats_buffer_max_fullness;
	float stats_alu_not_utilized     	= stats.stats_alu_not_utilized;
	float stats_total_cycles			= stats.stats_total_cycles;



    return std::make_tuple(res,stats_zero_ops,stats_1thread_mult_ops,stats_multi_thread_mult_ops,stats_buffer_fullness_acc,stats_buffer_max_fullness,stats_alu_not_utilized,stats_total_cycles);
}
inline std::tuple<xt::pyarray<int8_t>,float , float ,float, float ,float ,float ,float > run_int8(
	uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth, xt::pyarray<int8_t> &a, xt::pyarray<int8_t> &b,
	bool push_back_en, bool low_prec_mult_en, bool run_parallel)
{
	int bits = sizeof(uint8_t)*8;
	max_number = pow2(bits)-1;
	max_number_half_bits = pow2(bits/2);
	signed_max_number = pow2(bits)/2-1;
	signed_min_number = -pow2(bits)/2;
	
	
	stats_str stats;
	_node_push_back_en = push_back_en;
	_node_low_prec_mult_en = low_prec_mult_en;
	_run_parallel = run_parallel;
    smt_sa_os<int8_t> sa(dim, threads,alu_num, max_depth);
    sa.set_inputs(a, b);
	auto start = high_resolution_clock::now();
	xt::pyarray<int8_t> res = sa.go(stats);
    auto end = high_resolution_clock::now();
	cout<<"$$$$ Total test runtime   :   "<<duration_cast<microseconds>(end-start).count()<<"[us]"<<endl;

	float stats_zero_ops             	= stats.stats_zero_ops;
	float stats_1thread_mult_ops     	= stats.stats_1thread_mult_ops;
	float stats_multi_thread_mult_ops	= stats.stats_multi_thread_mult_ops;
	float stats_buffer_fullness_acc  	= stats.stats_buffer_fullness_acc;
	float stats_buffer_max_fullness  	= stats.stats_buffer_max_fullness;
	float stats_alu_not_utilized     	= stats.stats_alu_not_utilized;
	float stats_total_cycles			= stats.stats_total_cycles;



    return std::make_tuple(res,stats_zero_ops,stats_1thread_mult_ops,stats_multi_thread_mult_ops,stats_buffer_fullness_acc,stats_buffer_max_fullness,stats_alu_not_utilized,stats_total_cycles);
}

//inline xt::pyarray<float> run_fp32(uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth, xt::pyarray<float> &a, xt::pyarray<float> &b)
//{
//    smt_sa_os<float> sa(dim, threads,alu_num, max_depth);
//    sa.set_inputs(a, b);
//    return sa.go();
//}
//
//inline int test_example1(int x){
//    if(x > 0){
//        return 7;
//    }else{
//        return 3;
//    }
//}
//



PYBIND11_MODULE(cpy_smt_sa, m)
{
    xt::import_numpy();
    m.doc() = "Binding to C++ implementation of SMT-SA";

    m.def("run_uint8", run_uint8, "Execute the SMT-SA-OS uint8");
	m.def("run_int8", run_int8, "Execute the SMT-SA-OS int8");
	//m.def("foo", [](int i) { int rv = foo(i); return std::make_tuple(rv, i); });
   // m.def("run_fp32", run_fp32, "Execute the SMT-SA-OS FP32");

    //m.def("test_example1", test_example1, "dummy example of how to bind functions");
}
