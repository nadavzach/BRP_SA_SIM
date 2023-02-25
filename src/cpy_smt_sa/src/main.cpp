#include "pybind11/pybind11.h"

#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include "smt_sa_os.cpp"

#include <iostream>

namespace py = pybind11;


inline xt::pyarray<int32_t> run_int32(uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth, xt::pyarray<int32_t> &a, xt::pyarray<int32_t> &b)
{
    smt_sa_os<int32_t> sa(dim, threads,alu_num, max_depth);
    sa.set_inputs(a, b);
    return sa.go();
}

inline xt::pyarray<float> run_fp32(uint16_t dim, uint8_t threads,uint8_t alu_num, uint16_t max_depth, xt::pyarray<float> &a, xt::pyarray<float> &b)
{
    smt_sa_os<float> sa(dim, threads,alu_num, max_depth);
    sa.set_inputs(a, b);
    return sa.go();
}

inline int test_example1(int x){
    if(x > 0){
        return 7;
    }else{
        return 3;
    }
}




PYBIND11_MODULE(cpy_smt_sa, m)
{
    xt::import_numpy();
    m.doc() = "Binding to C++ implementation of SMT-SA";

    m.def("run_int32", run_int32, "Execute the SMT-SA-OS INT32");
    m.def("run_fp32", run_fp32, "Execute the SMT-SA-OS FP32");

    m.def("test_example1", test_example1, "dummy example of how to bind functions");
}
