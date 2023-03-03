#include "pybind11/pybind11.h"

#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include "smt_sa_os.cpp"

#include <iostream>

namespace py = pybind11;

inline xt::pyarray<int32_t> run_int32(uint16_t dim, uint8_t threads, uint16_t max_depth, xt::pyarray<int32_t> &a, xt::pyarray<int32_t> &b)
{
    smt_sa_os<int32_t> sa(dim, threads, max_depth);
    sa.set_inputs(a, b);
    return sa.go();
}



PYBIND11_MODULE(baseline_smt_sa, m)
{
    xt::import_numpy();
    m.doc() = "Binding to C++ implementation of SMT-SA";

    m.def("run_int32", run_int32, "Execute the SMT-SA-OS int32");
}
