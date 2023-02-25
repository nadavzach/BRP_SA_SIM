#include <iostream>
#include <cassert>
#include "xtensor/xarray.hpp"
#include "smt_sa_os.cpp"

using namespace std;

template <typename T>
xt::xarray<T> matmul(xt::xarray<T> a, xt::xarray<T> b) {
    xt::xarray<T> out = xt::zeros<T>({a.shape()[0], a.shape()[1], b.shape()[1]});

    for (uint32_t i=0; i<a.shape()[0]; i++) {
        for (uint32_t j=0; j<a.shape()[1]; j++) {
            for (uint32_t k=0; k<b.shape()[1]; k++) {
                for (uint32_t q=0; q<a.shape()[2]; q++) {
                    out(i,j,k) += a(i,j,q) * b(q,k);
                }
            }
        }
    }

    return out;
}

int main()
{
    //grid<float> sa(2,1);
    smt_sa_os<float> sa_os(16, 2, 4);

    //xt::xarray<float> a = xt::ones<int>({1, 3, 4});
    //xt::xarray<float> b = xt::ones<int>({4, 2});
    //a(0,0,0) = 1;
    //a(0,0,1) = 2;
    //a(0,0,2) = 3;
    //a(0,1,0) = 4;
    //a(0,1,1) = 5;
    //a(0,1,2) = 6;
    //a(1,0,0) = 7;
    //a(1,0,1) = 8;
    //a(1,1,0) = 9;
    //a(1,1,1) = 10;
    //b(0,0) = 11;
    //b(0,1) = 12;
    //b(1,0) = 13;
    //b(1,1) = 14;
    //b(2,0) = 15;
    //b(2,1) = 16;

    xt::xarray<float> a = xt::random::randn<float>({2, 33, 251});
    xt::xarray<float> b = xt::random::randn<float>({251, 29});
    //xt::xarray<int> a = xt::random::randint<int>({4, 128, 512}, -64, 64);
    //xt::xarray<int> b = xt::random::randint<int>({512, 128}, -64, 64);

    sa_os.set_inputs(a, b);
    xt::xarray<float> result = sa_os.go();
    cout << "Systolic array simulation result" << endl;
    cout << result << endl;

    xt::xarray<float> result_ref = matmul<float>(a, b);
    cout << endl << "Matrix multiplication result" << endl;
    cout << result_ref << endl;

    assert(xt::sum(result - result_ref)[0] == 0);
    cout << endl << "Passed!" << endl;


    return 0;
}
