#include "pybind11/pybind11.h"

#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include "smt_sa_os.cpp"

#include <iostream>

namespace py = pybind11;

inline xt::pyarray<uint8_t> run_uint8(uint16_t dim, uint8_t threads, uint16_t max_depth, xt::pyarray<uint8_t> &a, xt::pyarray<uint8_t> &b)
{
    smt_sa_os<uint8_t> sa(dim, threads, max_depth);
    sa.set_inputs(a, b);
    return sa.go();
}



PYBIND11_MODULE(baseline_smt_sa, m)
{
    xt::import_numpy();
    m.doc() = "Binding to C++ implementation of SMT-SA";

    m.def("run_uint8", run_uint8, "Execute the SMT-SA-OS uint8");
}
