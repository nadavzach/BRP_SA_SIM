import cpy_smt_sa as m
from unittest import TestCase
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text

def dequantize_to_float32(x,x_delta,y_delta):
            # Convert uint8 values back to float32 and scale to 0-1 range
            x = np.array(x, dtype=np.float32)
            x = x * x_delta * y_delta
            return x
def uniform_quantization_a(x):
        bits = 8
        x_max = np.max(x)
        N = 2**bits
        delta = x_max / N
        x_int = np.round(x / delta)
        x_q = np.clip(x_int, 0, N - 1)
        return x_q, delta
def uniform_quantization_b(x):
        bits = 8
        x_max = np.max(x)
        x_min = np.min(x)
        N = 2**bits
        delta = max(abs(x_min), abs(x_max)) * 2 / N
        #delta = x_max / N
        x_int = np.round(x / delta)
        x_q = np.clip(x_int, -N/2, N/2 - 1)
        return x_q, delta

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
            test_output_tuples_list =[]
            print("SA configuration:\n")
            print(" a is of size ("+str(a_w)+" x "+str(a_h)+" x "+str(a_c)+")\n")
            print(" b is of size ("+str(b_w)+" x "+str(b_h)+")\n")
            print(" the systolic array is of size (" +str(dim) +" x " + str(dim)+")\n ")
            print("\n\n\n   --- start running tests from default configurations list ---   \n\n\n")
            a = np.random.uniform(low=-20.0,high=20.0,size = (a_w,a_h,a_c)).astype(np.float32)
            b = np.random.uniform(low=-20.0,high=20.0,size = (b_w,b_h)).astype(np.float32)
            
            # This is for Python to print floats in a readable way
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)

            a_num_zeros = int(zero_per * a_w * a_h * a_c)
            a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
            a.ravel()[a_zero_indices] = 0
            b_num_zeros = int(zero_per * b_w * b_h )
            b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
            b.ravel()[b_zero_indices] = 0
            
            a_uint8, a_delta = uniform_quantization_a(a)
            b_int8,b_delta = uniform_quantization_b(b)
            base_line_test_output = m.run_int8(dim,1,1,1000,a_uint8,b_int8,True,False,False)
            for test_config in test_configs_list:
                max_depth = test_config[0]
                threads = test_config[1]
                alu_num = test_config[2]
                enable_pushback = test_config[3]
                enable_low_prec_mult = test_config[4]
                #if(not enable_low_prec_mult and not enable_pushback):#not supported ?
                #   continue 
                dont_run_1 = threads == 4 and alu_num <2
                if(dont_run_1):#not supported ?
                   continue 
                result_tuple = m.run_int8(dim,threads,alu_num,max_depth,a_uint8,b_int8,enable_pushback,enable_low_prec_mult,run_parallel)
                dequant_res = dequantize_to_float32(result_tuple[0],a_delta,b_delta)
                dequant_res_baseline = dequantize_to_float32(base_line_test_output[0],a_delta,b_delta)
                

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
                area_calc = 1#TODO
                stats_alu_total = stats_total_cycles * dim*dim*alu_num
                alu_utilized = 100*(stats_1thread_mult_ops + stats_multi_thread_mult_ops )/stats_alu_total

                all_result_tuple = tuple((stats_zero_ops,stats_1thread_mult_ops,stats_multi_thread_mult_ops,stats_buffer_fullness_acc,stats_buffer_max_fullness,stats_alu_not_utilized,stats_total_cycles,stats_speed_up,stats_ops_total,mse_from_base_line,area_calc,alu_utilized))
                test_output_tuples_list.append(tuple((test_config,all_result_tuple)))
            create_excel_table(test_output_tuples_list, "pre_saved_configs_test_outputs", './src/cpy_smt_sa/tests/results/')

            #mse_acc_spd_up_data = []
            #one_thread_diff_buff_mse_data = []
            #two_thread_diff_buff_mse_data = []
            #four_thread_diff_buff_mse_data = []
            #one_thread_diff_buff_su_data = []
            #two_thread_diff_buff_su_data = []
            #four_thread_diff_buff_su_data = []

            #creating for each thread config a speed-up - mse_Acc graph
            #for test_output in test_output_tuples_list:
            #    
            #    test_config = test_output[0]
            #    result_tuple = test_output[1]
            #    dequant_res = dequantize_to_float32(result_tuple[0])

            #    alu_num = test_config[2]
            #    buff_size = test_config[0]
            #    num_of_threads = test_config[1]
            #    enable_pushback = test_config[3]
            #    enable_low_prec_mult = test_config[4]
            #    #calc statistics:
            #    stats_zero_ops              = result_tuple[1]
            #    stats_1thread_mult_ops      = result_tuple[2]
            #    stats_multi_thread_mult_ops = result_tuple[3]
            #    stats_buffer_fullness_acc   = result_tuple[4]
            #    stats_buffer_max_fullness   = result_tuple[5]
            #    stats_alu_not_utilized      = result_tuple[6]
            #    stats_total_cycles          = result_tuple[7]
            #    stats_speed_up = base_line_test_output[7] / stats_total_cycles
            #    stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;
            #    mse_from_base_line = np.mean((dequant_res-base_line_test_output[0])**2)
            #    area_calc = 1#TODO
            #    stats_alu_total = stats_total_cycles * dim*dim*alu_num
            #    alu_utilized = 100*(stats_1thread_mult_ops + stats_multi_thread_mult_ops )/stats_alu_total


            #    mse_acc_spd_up_data.append(tuple((stats_speed_up,mse_from_base_line,("("+str(test_config[0])+","+str(test_config[1])+","+str(test_config[2])+")"))))
            #    
            #    if(enable_low_prec_mult and enable_pushback):
            #        if(num_of_threads == 1):
            #            one_thread_diff_buff_mse_data.append(tuple((buff_size,mse_from_base_line,("( area= "+str(area_calc) + ", alu util ="+str(alu_utilized)+" )"))))
            #            one_thread_diff_buff_su_data.append(tuple((buff_size,stats_speed_up,("( area= "+str(area_calc) + ", alu util ="+str(alu_utilized)+" )"))))
            #        elif num_of_threads == 2:
            #            two_thread_diff_buff_mse_data.append(tuple((buff_size,mse_from_base_line,("( area= "+str(area_calc) + ", alu util ="+str(alu_utilized)+" )"))))
            #            two_thread_diff_buff_su_data.append(tuple((buff_size,stats_speed_up,("( area= "+str(area_calc) + ", alu util ="+str(alu_utilized)+" )"))))
            #        elif num_of_threads == 4:
            #            four_thread_diff_buff_mse_data.append(tuple((buff_size,mse_from_base_line,("( area= "+str(area_calc) + ", alu util ="+str(alu_utilized)+", alu num="+str(alu_num)+" )"))))
            #            four_thread_diff_buff_su_data.append(tuple((buff_size,stats_speed_up,("( area= "+str(area_calc) + ", alu util ="+str(alu_utilized)+", alu num="+str(alu_num)+" )"))))



            #plot_data(mse_acc_spd_up_data,"speed ip from base line","mse from base line","speed_up_mse__all_configs_graph",True,'labels - (buffer depth,threads,alu num)')

            #plot_data(one_thread_diff_buff_mse_data,"buffer size","mse from base line","one_thread_diff_buffs_mse_graph")
            #plot_data(two_thread_diff_buff_mse_data,"buffer size","mse from base line","two_thread_diff_buffs_mse_graph")
            #plot_data(four_thread_diff_buff_mse_data,"buffer size","mse from base line","four_thread_diff_buffs_mse_graph")

            #plot_data(one_thread_diff_buff_su_data,"buffer size","speed_up from base line","one_thread_diff_buffs_speed_up_graph")
            #plot_data(two_thread_diff_buff_su_data,"buffer size","speed_up from base line","two_thread_diff_buffs_speed_up_graph")
            #plot_data(four_thread_diff_buff_su_data,"buffer size","speed_up from base line","four_thread_diff_buffs_speed_up_graph")

                

        else:

            a = np.random.uniform(low=-50.0,high=50.0,size = (a_w,a_h,a_c)).astype(np.float32)
            b = np.random.uniform(low=-50.0,high=50.0,size = (b_w,b_h)).astype(np.float32)
            
            # This is for Python to print floats in a readable way
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)
            
            a_num_zeros = int(zero_per * a_w * a_h * a_c)
            a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
            a.ravel()[a_zero_indices] = 0
            b_num_zeros = int(zero_per * b_w * b_h )
            b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
            b.ravel()[b_zero_indices] = 0
            a_uint8, a_delta = uniform_quantization_a(a)
            b_int8,b_delta = uniform_quantization_b(b)
            print("running base line test... \n\n")
            base_line_test_output = m.run_int8(dim,1,1,1000,a_uint8,b_int8,True,False,False)
            print("running test for configuration: buffer size= " + str(max_depth) + ", threads num= "+ str(threads) + ", alu num= " + str(alu_num) + ", push back= " + str(enable_pushback) + ", low precision mult= " +str(enable_low_prec_mult)+" \n")
            result_tuple = m.run_int8(dim,threads,alu_num,max_depth,a_uint8,b_int8,enable_pushback,enable_low_prec_mult,run_parallel)
            
            dequant_res = dequantize_to_float32(result_tuple[0],a_delta,b_delta)
            dequant_res_baseline = dequantize_to_float32(base_line_test_output[0],a_delta,b_delta)
            #calc statistics:
            stats_zero_ops              = result_tuple[1]
            stats_1thread_mult_ops      = result_tuple[2]
            stats_multi_thread_mult_ops = result_tuple[3]
            stats_buffer_fullness_acc   = result_tuple[4]
            stats_buffer_max_fullness   = result_tuple[5]
            stats_alu_not_utilized      = result_tuple[6]
            stats_total_cycles          = result_tuple[7]
            stats_speed_up = base_line_test_output[7] / stats_total_cycles
            stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;
            diff = (np.abs(dequant_res-dequant_res_baseline))
            diff_p2 = np.square(diff)
            mse_from_base_line = np.mean(diff_p2)
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



def plot_data(data,x_label,y_label,fig_save_name,gen_text = False,textstr=""):
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
    fig.set_size_inches(20,20)
    ax.scatter(x_vals, y_vals)
    
    # Add labels to each point
    texts = []
    for i, label in enumerate(labels):
        texts.append(plt.text(x_vals[i], y_vals[i],label))

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add general text to the graph
    if(gen_text):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    
    # Show the plot
    #plt.show()
    path =  './src/cpy_smt_sa/tests/results/'+str(fig_save_name)+'_graph.jpeg'
    if(os.path.exists(path)):
        os.remove(path)
    plt.savefig(path)

def create_excel_table(test_output_tuples_list, filename, path):

    df = pd.DataFrame(columns= range(len(test_output_tuples_list[0][1])+len(test_output_tuples_list[0][0])))
    header = tuple(("max_depth","threads","alu_num","pushback","low_prec_mult","zero_ops","1thread_mult_ops","multi_thread_mult_ops","buffer_fullness_acc","buffer_max_fullness","alu_not_utilized","total_cycles","speed_up","ops_total","mse_from_base_line","area_calc","alu_utilized"))
    df.loc[0] = header
    i =1
    for j, (config, result) in enumerate(test_output_tuples_list):
        num_nans = len(df.columns) - len(config)
        data_to_add = config + result#(np.nan,) * num_nans
        df.loc[i] = data_to_add
        #df.loc[i+1] = result
        i=i+1
    
    writer = pd.ExcelWriter(path + filename + '.xlsx')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unittest with an expected value')
    parser.add_argument('--threads',default=2, type=int, help='number of threads')
    parser.add_argument('--dim',default=3, type=int, help='number of dim')
    parser.add_argument('--alu_num',default=1, type=int, help='number of alu units')
    parser.add_argument('--buffer_size',default=5, type=int, help='buffer size')
    parser.add_argument('--enable_pushback',default=True, type=int, help='enable push back')
    parser.add_argument('--enable_low_prec_mult',default=True, type=int, help='enable low precision multiplication on alu')
    parser.add_argument('--run_pre_saved_configs', type=int, help='run hard coded simulations for number of threads (1,2,4),buffer size (1,5,10,20,50,100), pushback(true,false)')
    parser.add_argument('--a_w', type=int,default=20, help='a width')
    parser.add_argument('--a_h', type=int,default=20, help='a height')
    parser.add_argument('--a_c', type=int,default=20, help='a channels')
    parser.add_argument('--b_h', type=int,default=20, help='b height')
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