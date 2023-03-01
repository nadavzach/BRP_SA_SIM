import cpy_smt_sa as m
from unittest import TestCase
import unittest
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import os

def quantize_to_uint8(x):
            # Scale float32 values to 0-255 range and convert to uint8
            x = np.clip(x, 0, 255)
            x = np.round(x).astype(np.uint8)
            return x

def dequantize_to_float32(x):
            # Convert uint8 values back to float32 and scale to 0-1 range
            x = np.array(x, dtype=np.float32)
            x /= 255.0
            return x


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
        run_parallel = self.run_parallel
        
        a_w = self.a_w
        a_h = self.a_h 
        a_c = self.a_c 
        
        b_w = a_c 
        b_h = self.b_h 

        zero_per = self.zero_per/100

        

        if(run_pre_saved_configs):
            max_depth_opts = [1,5,10,20]
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
            test_output_tuples_list =[]
            print("SA configuration:\n")
            print(" a is of size ("+str(a_w)+" x "+str(a_h)+" x "+str(a_c)+")\n")
            print(" b is of size ("+str(b_w)+" x "+str(b_h)+")\n")
            print(" the systolic array is of size (" +str(dim) +" x " + str(dim)+")\n ")
            print("\n\n\n   --- start running tests from default configurations list ---   \n\n\n")


            a = np.random.uniform(low=-20.0,high=20.0,size = (a_w,a_h,a_d)).astype(np.float32)
            b = np.random.uniform(low=-20.0,high=20.0,size = (b_w,b_h)).astype(np.float32)

            a_num_zeros = int(zero_per * a_w * a_h * a_c)
            a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
            a.ravel()[a_zero_indices] = 0
            b_num_zeros = int(zero_per * b_w * b_h )
            b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
            b.ravel()[b_zero_indices] = 0

            a_int8 = quantize_to_uint8(a)
            b_int8 = quantize_to_uint8(b)
            base_line_test_output = m.run_int8(dim,1,1,1000,a_int8,b_int8,True,False,False)
            for test_config in test_configs_list:
                max_depth = test_config[0]
                threads = test_config[1]
                alu_num = test_config[2]
                enable_pushback = test_config[3]
                enable_low_prec_mult = test_config[4]
                #if(not enable_low_prec_mult and not enable_pushback):#not supported ?
                #   continue 
                result_tuple = m.run_int8(dim,threads,alu_num,max_depth,a_int8,b_int8,enable_pushback,enable_low_prec_mult,run_parallel)
                dequant_res = dequantize_to_float32(result_tuple[0])
                dequant_res_baseline = dequantize_to_float32(base_line_test_output[0])
                mse_from_base_line = np.mean((dequant_res-dequant_res_baseline)**2)
                test_output_tuples_list.append(tuple((test_config,result_tuple)))

            mse_cycles_data = []
            for test_output in test_output_tuples_list:
                
                test_config = test_output[0]
                result_tuple = test_output[1]
                dequant_res = dequantize_to_float32(result_tuple[0])

                #calc statistics:
                stats_zero_ops              = result_tuple[1]
                stats_1thread_mult_ops      = result_tuple[2]
                stats_multi_thread_mult_ops = result_tuple[3]
                stats_buffer_fullness_acc   = result_tuple[4]
                stats_buffer_max_fullness   = result_tuple[5]
                stats_alu_not_utilized      = result_tuple[6]
                stats_total_cycles          = result_tuple[7]
                stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;
                mse_from_base_line = np.mean((dequant_res-dequant_res_baseline)**2)
                print(mse_from_base_line)
                print(test_config)

                mse_cycles_data.append(tuple((stats_total_cycles,mse_from_base_line,("("+str(test_config[0])+","+str(test_config[1])+","+str(test_config[2])+","+str(test_config[3])+","+str(test_config[4])+")"))))
            plot_data(mse_cycles_data,"Cycles","mse")
        else:

            a = np.random.randint(0,5,size = (a_w,a_h,a_c))
            b = np.random.randint(0,5,size = (b_w,b_h))

            a_num_zeros = int(zero_per * a_w * a_h * a_c)
            a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
            a.ravel()[a_zero_indices] = 0
            b_num_zeros = int(zero_per * b_w * b_h )
            b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
            b.ravel()[b_zero_indices] = 0
            a_int8 = quantize_to_uint8(a)
            b_int8 = quantize_to_uint8(b)
            print("running base line test... \n\n")
            base_line_test_output = m.run_int8(dim,1,1,1000,a_int8,b_int8,True,False,False)
            print("running test for configuration: buffer size= " + str(max_depth) + ", threads num= "+ str(threads) + ", alu num= " + str(alu_num) + ", push back= " + str(enable_pushback) + ", low precision mult= " +str(enable_low_prec_mult)+" \n")
            result_tuple = m.run_int8(dim,threads,alu_num,max_depth,a_int8,b_int8,enable_pushback,enable_low_prec_mult,run_parallel)
            
            dequant_res_baseline = dequantize_to_float32(base_line_test_output[0])
            #calc statistics:
            dequant_res                 = dequantize_to_float32(result_tuple[0])
            stats_zero_ops              = result_tuple[1]
            stats_1thread_mult_ops      = result_tuple[2]
            stats_multi_thread_mult_ops = result_tuple[3]
            stats_buffer_fullness_acc   = result_tuple[4]
            stats_buffer_max_fullness   = result_tuple[5]
            stats_alu_not_utilized      = result_tuple[6]
            stats_total_cycles          = result_tuple[7]
            stats_speed_up = base_line_test_output[7] / stats_total_cycles
            stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;
            mse_from_base_line = np.mean((dequant_res-dequant_res_baseline)**2)
            stats_alu_total = stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops + stats_alu_not_utilized

            print("finished test, result - \n")
            print(dequant_res)
            print("total cycles                     :  " +str(stats_total_cycles))
            print("speed up from base line          :  " +str(stats_speed_up))
            print("stats_zero_ops %                 :  " +str(100*stats_zero_ops/stats_ops_total             ))
            print("stats_1thread_mult_ops %         :  " +str(100*stats_1thread_mult_ops/stats_ops_total     ))
            print("stats_multi_thread_mult_ops %    :  " +str(100*threads*stats_multi_thread_mult_ops/stats_ops_total ))
            print("stats_total_thread_mult_ops %    :  " +str(stats_ops_total ))
            print("stats_buffer_fullness_acc        :  " +str(stats_buffer_fullness_acc  ))
            print("stats_buffer_max_fullness        :  " +str(stats_buffer_max_fullness  ))
            print("MSE from base line               :  " +str(mse_from_base_line  ))
            print("stats_alu_not_utilized %         :  " +str(100*stats_alu_not_utilized/stats_alu_total ))



def plot_data(data,x_label,y_label):
    # Unpack the data into separate arrays for x, y, and label
    x_vals = []   
    y_vals = []
    labels = []

    for d in data:
        x_vals.append(d[0]) 
        y_vals.append(d[1]) 
        labels.append(d[2]) 

    # Create the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(30,30)
    ax.scatter(x_vals, y_vals)
    
    # Add labels to each point
    for i, label in enumerate(labels):
        ax.annotate(label, (x_vals[i], y_vals[i])).set_fontsize(20)

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add general text to the graph
    textstr = 'labels - (buffer depth,threads,alu num,enable pushback,enable low prec mult)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    # Show the plot
    #plt.show()
    path =  './src/cpy_smt_sa/tests/results/'+str(x_label)+'_'+str(y_label)+'_graph.jpeg'
    if(os.path.exists(path)):
        os.remove(path)
    plt.savefig(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unittest with an expected value')
    parser.add_argument('--threads',default=1, type=int, help='number of threads')
    parser.add_argument('--dim',default=3, type=int, help='number of dim')
    parser.add_argument('--alu_num',default=1, type=int, help='number of alu units')
    parser.add_argument('--buffer_size',default=10, type=int, help='buffer size')
    parser.add_argument('--enable_pushback',default=True, type=int, help='enable push back')
    parser.add_argument('--enable_low_prec_mult',default=True, type=int, help='enable low precision multiplication on alu')
    parser.add_argument('--run_pre_saved_configs', type=int, help='run hard coded simulations for number of threads (1,2,4),buffer size (1,5,10,20,50,100), pushback(true,false)')
    parser.add_argument('--a_w', type=int,default=5, help='a width')
    parser.add_argument('--a_h', type=int,default=5, help='a height')
    parser.add_argument('--a_c', type=int,default=5, help='a channels')
    parser.add_argument('--b_h', type=int,default=5, help='b height')
    parser.add_argument('--zero_per', type=int,default=10, help='% of zeros in a and b arrays - o to 100')
    parser.add_argument('--run_parallel', action='store_true',
                    help='run simulation on multiple OS threads (default: False)', default=False)

    args = parser.parse_args()
    testSa_obj = TestSa()
    testSa_obj.threads = args.threads
    testSa_obj.dim = args.dim
    testSa_obj.alu_num = args.alu_num
    testSa_obj.buffer_size = args.buffer_size
    testSa_obj.enable_pushback = args.enable_pushback
    testSa_obj.enable_low_prec_mult = args.enable_low_prec_mult
    testSa_obj.run_pre_saved_configs = args.run_pre_saved_configs
    testSa_obj.a_w = args.a_w
    testSa_obj.a_h = args.a_h
    testSa_obj.a_c = args.a_c
    testSa_obj.b_h = args.b_h
    testSa_obj.zero_per = args.zero_per
    testSa_obj.run_parallel = args.run_parallel
    #unittest.main()
    testSa_obj.test_brp_sa()

    
