import cpy_smt_sa as m
from unittest import TestCase
import numpy as np
import random
import argparse



class TestSa(TestCase):

   
    def test_brp_sa(self):
        stats_zero_ops = 0
        stats_1thread_mult_ops = 0
        stats_multi_thread_mult_ops = 0
        stats_alu_not_utilized = 0
        stats_buffer_fullness_acc = 0
        stats_buffer_max_fullness = 0
        
        np.random.seed(0)
        dim = self.dim
        threads = self.threads
        alu_num = self.alu_num
        max_depth = self.buffer_size 
        enable_pushback      = self.enable_pushback
        enable_low_prec_mult = self.enable_low_prec_mult
        run_pre_saved_configs= self.run_pre_saved_configs
        
        a_w = self.a_w
        a_h = self.a_h 
        a_c = self.a_c 
        
        b_w = a_c 
        b_h = self.b_h 

        per = self.per/100

        if(run_pre_saved_configs):
            max_depth_opts = [1,5,10,20,50,100]
            threads_opts = [1,2,4]
            alu_num_opts = [1,2]
            pushback_opts = [True,False]
            low_perc_mult_opts = [True,False]
            arr_list = [max_depth_opts,threads_opts,alu_num_opts,pushback_opts,low_perc_mult_opts]

            # Initialize an empty list to store the output tuples
            test_configs_list = []

            # Use NumPy's `meshgrid` function to create a grid of indices for each input array
            grid = np.meshgrid(*[np.arange(len(arr)) for arr in arr_list])

            # Use NumPy's `stack` function to combine the indices into a single 2D array
            indices = np.stack(grid, axis=-1).reshape(-1, len(arr_list))

            # Iterate over the indices and create a tuple with the corresponding values from each input array
            for index in indices:
                test_configs_list.append(tuple(arr[index[i]] for i, arr in enumerate(arr_list)))
            for test_config in test_configs_list:

                max_depth = test_config[0]
                threads = test_config[1]
                alu_num = test_config[2]
                enable_pushback = test_config[3]
                enable_low_prec_mult = test_config[4]
                
                a = np.random.randint(0,255,size = (a_w,a_h,a_c))
                b = np.random.randint(0,255,size = (b_w,b_h))

                a_num_zeros = int(per * a_w * a_h * a_c)
                a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
                a.ravel()[a_zero_indices] = 0
                b_num_zeros = int(per * b_w * b_h )
                b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
                b.ravel()[b_zero_indices] = 0


                result_tuple = m.run_uint8(dim,threads,alu_num,max_depth,a,b,stats_alu_not_utilized, stats_zero_ops,stats_1thread_mult_ops,stats_multi_thread_mult_ops,stats_buffer_fullness_acc,stats_buffer_max_fullness)
                result = result_tuple[0]
                stats_zero_ops              = result_tuple[1]
                stats_1thread_mult_ops      = result_tuple[2]
                stats_multi_thread_mult_ops = result_tuple[3]
                stats_buffer_fullness_acc   = result_tuple[4]
                stats_buffer_max_fullness   = result_tuple[5]
                stats_alu_not_utilized      = result_tuple[6]

                stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;



        print("finished test, result - \n")
        print(result.astype(np.uint))
        print("stats_zero_ops %             :  " +str(100*stats_zero_ops/stats_ops_total             ))
        print("stats_1thread_mult_ops %     :  " +str(100*stats_1thread_mult_ops/stats_ops_total     ))
        print("stats_multi_thread_mult_ops % :  " +str(100*threads*stats_multi_thread_mult_ops/stats_ops_total ))
        print("stats_total_thread_mult_ops % :  " +str(stats_ops_total ))
        print("stats_alu_not_utilized     :  " +str(stats_alu_not_utilized     ))
        print("stats_buffer_fullness_acc  :  " +str(stats_buffer_fullness_acc  ))
        print("stats_buffer_max_fullness  :  " +str(stats_buffer_max_fullness  ))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unittest with an expected value')
    parser.add_argument('--threads', type=int, help='number of threads')
    parser.add_argument('--dim', type=int, help='number of dim')
    parser.add_argument('--alu_num', type=int, help='number of alu units')
    parser.add_argument('--buffer_size', type=int, help='buffer size')
    parser.add_argument('--enable_pushback', type=int, help='enable push back')
    parser.add_argument('--enable_low_prec_mult', type=int, help='enable low precision multiplication on alu')
    parser.add_argument('--run_pre_saved_configs', type=int, help='run hard coded simulations for number of threads (1,2,4),buffer size (1,5,10,20,50,100), pushback(true,false)')
    parser.add_argument('--a_w', type=int, help='a width'b
    parser.add_argument('--a_h', type=int, help='a height')
    parser.add_argument('--a_c', type=int, help='a channels')
    parser.add_argument('--b_h', type=int, help='b height')
    parser.add_argument('--zero_per', type=int,default=10, help='% of zeros in a and b arrays - o to 100')
    args = parser.parse_args()
    TestSa.threads = args.threads
    TestSa.dim = args.dim
    TestSa.alu_num = args.alu_num
    TestSa.buffer_size = args.buffer_size
    TestSa.enable_pushback = args.enable_pushback
    TestSa.enable_low_prec_mult = args.enable_low_prec_mult
    TestSa.run_pre_saved_configs = args.run_pre_saved_configs
    TestSa.a_w = args.a_w
    TestSa.a_h = args.a_h
    TestSa.a_c = args.a_c
    TestSa.b_h = args.b_h
    TestSa.zero_per = args.zero_per
    unittest.main()

    
