#include <iostream>
#include <vector>
#include <cmath>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xrandom.hpp"
#include "grid_os.cpp"
#include <omp.h>


using namespace std;

extern bool _run_parallel;

struct tile_idx {
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;
};

struct stats_str {
	float stats_zero_ops=0;
	float stats_1thread_mult_ops=0;
	float stats_multi_thread_mult_ops=0;
	float stats_alu_not_utilized=0;
	float stats_buffer_fullness_acc=0;
	float stats_buffer_max_fullness=0;
    float stats_total_cycles=0;

};


template <typename T>
class smt_sa_os {
private:
    uint16_t _dim;
    uint8_t _threads;
    uint8_t _alu_num;
    uint16_t _max_depth;
    xt::xarray<T> _a;
    xt::xarray<T> _b;

    void _subtile_dict(vector<uint16_t> &subtile_start, vector<uint16_t> &subtile_end);
    void _subtile_range(uint8_t thread, uint16_t &thread_tile_start, uint16_t &thread_tile_end);
public:
    grid<T> sa_grid;
    uint64_t cycles;
    uint64_t nodes_num=0;

    smt_sa_os (uint16_t dim, uint8_t threads, uint8_t alu_num, uint16_t max_depth=4096);

    void set_inputs(xt::xarray<T> a, xt::xarray<T> b);
    void get_tile(vector<xt::xarray<T>> &tile_a, vector<xt::xarray<T>> &tile_b, tile_idx t_idx);
    xt::xarray<T> go(stats_str& stats);
    xt::xarray<T> go(vector<tile_idx> &tile_vec,stats_str& stats);
};

template <typename T>
smt_sa_os<T>::smt_sa_os (uint16_t dim,uint8_t threads, uint8_t alu_num, uint16_t max_depth) : _dim(dim), _threads(threads),_alu_num(alu_num), _max_depth(max_depth), sa_grid(dim, threads,alu_num, max_depth), cycles(0) {}

template <typename T>
void smt_sa_os<T>::set_inputs(xt::xarray<T> a, xt::xarray<T> b) {
    assert(a.shape().size() == 3);
    assert(b.shape().size() == 2);
    assert(a.shape()[2] == b.shape()[0]);
    if(a.shape().size() != 3)
        cout<<"a size error!! size is: "<<a.shape().size()<<endl;
    if(b.shape().size() != 2)
        cout<<"b size error!! size is: "<<b.shape().size()<<endl;
    if(a.shape()[2] != b.shape()[0]){
        cout<<"a b size mismatch error!!"<<endl;
        cout<<"a.shape()[2] is: "<<a.shape()[2]<<endl;
        cout<<"b.shape()[0] is: "<<b.shape()[0]<<endl;
    }
    cout<<"a_w = "<<a.shape()[0]<<", a_h = "<<a.shape()[1]<<", a_c = "<<a.shape()[2]<<endl;
    cout<<"b_w = "<<b.shape()[0]<<", b_h = "<<b.shape()[1]<<endl;


    _a = a;
    _b = b; 
}

template <typename T>
void smt_sa_os<T>::get_tile(vector<xt::xarray<T>> &tile_a, vector<xt::xarray<T>> &tile_b, tile_idx t_idx) {
    uint16_t a_tile_H = _dim;
    uint16_t b_tile_W = _dim;

    vector<uint16_t> subtile_start;
    vector<uint16_t> subtile_end;
    _subtile_dict(subtile_start, subtile_end);

    for (uint8_t t=0; t<_threads; t++) {
        xt::xarray<T> tile = xt::view(_a, t_idx.d1, xt::range(t_idx.d2 * a_tile_H, (t_idx.d2+1)*a_tile_H), xt::range(subtile_start[t], subtile_end[t]));
        tile_a.push_back(tile);

        if (_dim > tile_a[t].shape()[0]) {
            tile_a[t] = xt::pad(tile_a[t], {{0, _dim - tile_a[t].shape()[0]}, {0, 0}});
        }
    }

    for (uint8_t t=0; t<_threads; t++) {
        xt::xarray<T> tile = xt::view(_b, xt::range(subtile_start[t], subtile_end[t]), xt::range(t_idx.d3*b_tile_W, (t_idx.d3+1)*b_tile_W));
        tile_b.push_back(tile);

        if (_dim > tile_b[t].shape()[1]) {
            tile_b[t] = xt::pad(tile_b[t], {{0, 0}, {0, _dim - tile_b[t].shape()[1]}});
        }
    }
}

template <typename T>
void smt_sa_os<T>::_subtile_dict(vector<uint16_t> &subtile_start, vector<uint16_t> &subtile_end) {
    for (uint8_t t=0; t<_threads; t++) {
        uint16_t start, end;
        _subtile_range(t, start, end);
        subtile_start.push_back(start);
        subtile_end.push_back(end);
    }
}

template <typename T>
void smt_sa_os<T>::_subtile_range(uint8_t thread, uint16_t &thread_tile_start, uint16_t &thread_tile_end) {
    uint16_t a_tile_W = _a.shape()[2];
    uint16_t b_tile_H = _b.shape()[0];
    assert(a_tile_W == b_tile_H);

    uint16_t subtile_size = floor(float(b_tile_H) / _threads);

    if (thread < _threads - 1) {
        thread_tile_start = thread * subtile_size;
        thread_tile_end = (thread+1) * subtile_size;
    }
    else {
        thread_tile_start = thread * subtile_size;
        thread_tile_end = b_tile_H;
    }
}
using namespace std;
using namespace std::chrono;
template <typename T>
xt::xarray<T> smt_sa_os<T>::go(vector<tile_idx> &tile_vec,stats_str& stats) {
    assert(tile_vec.size() > 0);
    uint16_t a_tiles = ceil(float(_a.shape()[1]) / _dim);
    uint16_t b_tiles = ceil(float(_b.shape()[1]) / _dim);

    // Assuming tile_vec is ordered (batch, height, width), i.e., rows->columns->depth!
    xt::xarray<uint32_t> array_ctrl = xt::ones<uint32_t>({_dim, _dim}) * ((tile_vec[0].d1 * a_tiles * b_tiles) + (tile_vec[0].d2 * b_tiles) + (tile_vec[0].d3));

    uint32_t global_tile_idx = 1;
    vector<xt::xarray<T>> tile_a, tile_b;
    get_tile(tile_a, tile_b, tile_vec[0]);
    for (uint8_t t=0; t<_threads; t++)
        sa_grid.push(tile_a[t], tile_b[t], t, true);

    xt::xarray<T> result = xt::zeros<T>({_a.shape()[0], _a.shape()[1], _b.shape()[1]});

    vector<uint16_t> subtile_start, subtile_end;
    _subtile_dict(subtile_start, subtile_end);

    uint32_t computed = 0;
    uint32_t while_end = tile_vec.size() * _dim * _dim;
    int sa_grid_cycle_time =0;
    int sa_for_loop_cycle_time =0;
    std::cout<<"  -- starting sagrid cycles --  \n\n";
    while (computed < while_end) {
        
        auto start = high_resolution_clock::now();
        sa_grid.cycle();
        auto end = high_resolution_clock::now();
        sa_grid_cycle_time += duration_cast<microseconds>(end-start).count();
        cycles++;
        auto start_for_loop = high_resolution_clock::now();
        if(_run_parallel){
            //cout<<"Running Parallel"<<endl;
            //#pragma omp parallel for
            for (uint16_t i=0; i<_dim; i++) {
                //#pragma omp parallel for
                for (uint16_t j=0; j<_dim; j++) {
                    uint8_t halt_count = 0;
            
                    for (uint8_t t=0; t<_threads; t++) {
                        if (sa_grid.nodes[i][j].is_halt(t)){
                            //#pragma omp atomic
                            halt_count++;
                        }

                        uint32_t acc_t = sa_grid.nodes[i][j].get_acc_t(t);
    //                    std::cout<<"thread: "<<(unsigned int)t<<" node: "<<i<<", "<<j<<" acc_t: "<<acc_t<<std::endl; // DEBUG
                        //Ssleep(1); // DEBUG
                        assert(subtile_end[t] - subtile_start[t] >= 0);

                        if ((acc_t == uint32_t(subtile_end[t] - subtile_start[t])) && !sa_grid.nodes[i][j].is_halt(t))
                            sa_grid.nodes[i][j].halt(t);
                    }
                    
                    if (halt_count == _threads) {
                        uint32_t batch = floor(float(array_ctrl(i, j)) / (a_tiles * b_tiles));
                        uint32_t i_result = int(i + int((array_ctrl(i, j) % (a_tiles * b_tiles)) / b_tiles) * _dim);
                        uint32_t j_result = int(j + ((array_ctrl(i, j) % (a_tiles * b_tiles)) % b_tiles) * _dim);

                        if (i_result < result.shape()[1] && j_result < result.shape()[2])
                            result(batch, i_result, j_result) = sa_grid.nodes[i][j].get_acc();
                        
                        array_ctrl(i, j)++;
                        sa_grid.nodes[i][j].reset_acc();
                        sa_grid.nodes[i][j].reset_acc_t();
                        //#pragma omp atomic
                        computed++;

                        sa_grid.nodes[i][j].release();
                    }
                    
                }	
            }
        }
        else{
            for (uint16_t i=0; i<_dim; i++) {
                for (uint16_t j=0; j<_dim; j++) {
                    uint8_t halt_count = 0;
            
                    for (uint8_t t=0; t<_threads; t++) {
                        if (sa_grid.nodes[i][j].is_halt(t))
                            halt_count++;

                        uint32_t acc_t = sa_grid.nodes[i][j].get_acc_t(t);
    //                    std::cout<<"thread: "<<(unsigned int)t<<" node: "<<i<<", "<<j<<" acc_t: "<<acc_t<<std::endl; // DEBUG
                        //Ssleep(1); // DEBUG
                        assert(subtile_end[t] - subtile_start[t] >= 0);

                        if ((acc_t == uint32_t(subtile_end[t] - subtile_start[t])) && !sa_grid.nodes[i][j].is_halt(t))
                            sa_grid.nodes[i][j].halt(t);
                    }
                    
                    if (halt_count == _threads) {
                        uint32_t batch = floor(float(array_ctrl(i, j)) / (a_tiles * b_tiles));
                        uint32_t i_result = int(i + int((array_ctrl(i, j) % (a_tiles * b_tiles)) / b_tiles) * _dim);
                        uint32_t j_result = int(j + ((array_ctrl(i, j) % (a_tiles * b_tiles)) % b_tiles) * _dim);

                        if (i_result < result.shape()[1] && j_result < result.shape()[2])
                            result(batch, i_result, j_result) = sa_grid.nodes[i][j].get_acc();
                        
                        array_ctrl(i, j)++;
                        sa_grid.nodes[i][j].reset_acc();
                        sa_grid.nodes[i][j].reset_acc_t();
                        computed++;

                        sa_grid.nodes[i][j].release();
                    }
                    
                }	
            }
        }

        auto end_for_loop = high_resolution_clock::now();
        sa_for_loop_cycle_time += duration_cast<microseconds>(end_for_loop - start_for_loop).count();

        if (sa_grid.mem_a[0]._buf[0].size() < 128) {
            if (global_tile_idx < tile_vec.size()) {
                tile_a.clear();
                tile_b.clear();

                get_tile(tile_a, tile_b, tile_vec[global_tile_idx]);

                for (uint8_t t=0; t<_threads; t++)
                    sa_grid.push(tile_a[t], tile_b[t], t);

                global_tile_idx++;
            }
        }

        /*if(cycles%500 == 0 && cycles > 1){
            cout<<"progress - "<<computed<<"/"<<while_end<<" -> "<<(100*computed/while_end)<<" % done ...\n";
            cout<<"cycles done = "<<cycles<<". \n\n";
            cout<<"grid_go_cycle avg time [us] = "<<(float)(sa_grid_cycle_time/cycles)<<"\n for loop avg time is: [us] "<<(float)sa_for_loop_cycle_time/cycles<<". \n";
            cout<<"total grid cycles [cycles]= "<< sa_grid_cycle_time/1000000<<" \n =============================== \n\n";
        }*/
    }
    cout<<"progress - "<<computed<<"/"<<while_end<<" -> "<<(100*computed/while_end)<<" % done ...\n";
    cout<<"cycles done = "<<cycles<<". \n\n";
    cout<<"grid_go_cycle avg time [us] = "<<(float)(sa_grid_cycle_time/cycles)<<"\n for loop avg time is: [us] "<<(float)sa_for_loop_cycle_time/cycles<<". \n";
    cout<<"total grid cycles [cycles]= "<< sa_grid_cycle_time/1000000<<" \n =============================== \n\n";

  	for (uint16_t i=0; i<_dim; i++) {
    	for (uint16_t j=0; j<_dim; j++) {

			//stats gather
			float stats_zero_ops;
			float stats_1thread_mult_ops;
			float stats_multi_thread_mult_ops;
			float stats_alu_not_utilized;
			float stats_buffer_fullness_acc;
			float stats_buffer_max_fullness;
		
			sa_grid.nodes[i][j].get_stats(stats_alu_not_utilized, stats_zero_ops,stats_1thread_mult_ops,stats_multi_thread_mult_ops,stats_buffer_fullness_acc,stats_buffer_max_fullness);
		
							
		
			stats.stats_zero_ops              += stats_zero_ops;
		    stats.stats_1thread_mult_ops      += stats_1thread_mult_ops;
		    stats.stats_multi_thread_mult_ops += stats_multi_thread_mult_ops;
		    stats.stats_alu_not_utilized      += stats_alu_not_utilized;
			if(i!=0 && j!=0){
		    	stats.stats_buffer_fullness_acc   += stats_buffer_fullness_acc;
				if( stats.stats_buffer_max_fullness < stats_buffer_max_fullness)
		    		stats.stats_buffer_max_fullness  = stats_buffer_max_fullness;
			}
			nodes_num++;
		}
	}


	//stats.stats_zero_ops              = stats.stats_zero_ops              / nodes_num;
    
    //stats.stats_1thread_mult_ops      = stats.stats_1thread_mult_ops      / nodes_num;
    //stats.stats_multi_thread_mult_ops = stats.stats_multi_thread_mult_ops / nodes_num;
    //stats.stats_alu_not_utilized      = stats.stats_alu_not_utilized      / nodes_num;
    stats.stats_buffer_fullness_acc   = stats.stats_buffer_fullness_acc   / nodes_num;
    stats.stats_total_cycles = cycles;

	//stats gather - end


    return result;
}



template <typename T>
xt::xarray<T> smt_sa_os<T>::go(stats_str& stats) {
    uint16_t batch = _a.shape()[0];
    uint16_t a_tiles = ceil(float(_a.shape()[1]) / _dim);
    uint16_t b_tiles = ceil(float(_b.shape()[1]) / _dim);
	

    xt::xarray<T> result = xt::zeros<T>({_a.shape()[0], _a.shape()[1], _b.shape()[1]});
    vector<tile_idx> tile_vec;

    for (uint16_t b=0; b<batch; b++) {
        for (uint16_t i=0; i<a_tiles; i++) {
            for (uint16_t j=0; j<b_tiles; j++) {
                tile_idx t_idx;
                t_idx.d1 = b;
                t_idx.d2 = i;
                t_idx.d3 = j;
                tile_vec.push_back(t_idx);
            }
        }
    }

    result += go(tile_vec,stats);


    return result;
}
