#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "fifo.cpp"
#include <cuda/atomic>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "node_pu_os.cu"
#include "node_mem.cu"

__global__ 
void cpp_process_grid_cycle_kernel(vector<vector<node_pu<T>>> nodes,uint16_t dim,uint8_t threads,uint8_t alu_num,uint16_t max_depth){
    process_grid_cycle_kernel<<<1,dim*dim>>(mem_a, mem_b, nodes,dim,threads,alu_num,max_depth);

}
__global__
void process_grid_cycle_kernel(vector<vector<node_pu<T>>> nodes,uint16_t dim,uint8_t threads,uint8_t alu_num,uint16_t max_depth){
    process_grid_cycle(mem_a, mem_b, nodes,dim,threads,alu_num,max_depth);
}
   uint16_t _dim;
    uint8_t _threads;
    uint8_t _alu_num;
    uint16_t _max_depth;

__device__ 
void process_grid_cycle(vector<vector<node_pu<T>>> nodes,uint16_t dim,uint8_t threads,uint8_t alu_num,uint16_t max_depth){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    nodes[tx][ty].go();
    __syncthreads()
  
    for (uint8_t t=0; t<_threads; t++) {
        nodes[tx][ty]._buf_a[t].cycle();
        nodes[tx][ty]._buf_b[t].cycle();
    }

    __syncthreads()

    if(ty == 0){
        mem_a[tx].go();
        mem_b[tx].go();

        for (uint8_t t=0; t<_threads; t++) {
            mem_a[tx]._buf[t].cycle();
            mem_b[tx]._buf[t].cycle();
        }
    }
}
