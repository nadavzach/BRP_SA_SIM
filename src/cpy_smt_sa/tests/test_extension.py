import cpy_smt_sa as m
import baseline_smt_sa as baseline
from unittest import TestCase
import numpy as np
import random
import time

def highlight_differences(arr1, arr2):
    """Highlight differences between two numpy arrays."""
    # Check that the input arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")

    # Create a boolean array indicating where the two arrays differ
    diff_mask = arr1 != arr2

    # Create a copy of the first array and set the differing elements to a different value
    highlight_arr = arr1.copy().astype(int)
    highlight_arr[diff_mask] = -1

    # Return the highlighted array
    return highlight_arr

class ExampleTest(TestCase):

    def test_example1(self):
        stats_zero_ops = 0
        stats_1thread_mult_ops = 0
        stats_multi_thread_mult_ops = 0
        stats_alu_not_utilized = 0
        stats_buffer_fullness_acc = 0
        stats_buffer_max_fullness = 0
        stats_total_cycles = 0
        
        np.random.seed(0)
        dim = 3
        threads = 2
        alu_num = 1
        max_depth = 2
        push_back = True
        low_prec_mult = True
        
        a_w = 3
        a_h = 3
        a_d = 3 
        
        b_w = a_d
        b_h = 3
        a = np.random.uniform(low=-20.0,high=20.0,size = (a_w,a_h,a_d)).astype(np.float32)
        b = np.random.uniform(low=-20.0,high=20.0,size = (b_w,b_h)).astype(np.float32)
        
        a_mask = np.random.randint(0,2,size = (a_w,a_h,a_d))
        b_mask = np.random.randint(0,2,size = (b_w,b_h))
        a=np.multiply(a,a_mask)
        b=np.multiply(b,b_mask)

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

    # Example usage with a 3D numpy array
        a_int8 = quantize_to_uint8(a)
        b_int8 = quantize_to_uint8(b)
        print("a array: \n")
        print(a)
        print("a quant: \n")
        print(a_int8)
        #print("b array: \n")
        #print(b)
        #print("running first test with run_int8:")
        result_tuple = m.run_int8(dim,threads,alu_num,max_depth,a_int8,b_int8,push_back,low_prec_mult,False)
        result = result_tuple[0]
        stats_zero_ops              = result_tuple[1]
        stats_1thread_mult_ops      = result_tuple[2]
        stats_multi_thread_mult_ops = result_tuple[3]
        stats_buffer_fullness_acc   = result_tuple[4]
        stats_buffer_max_fullness   = result_tuple[5]
        stats_alu_not_utilized      = result_tuple[6]
        stats_total_cycles          = result_tuple[7]

        stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops
        stats_alu_total = stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops + stats_alu_not_utilized
        #baseline_res = baseline.run_int8(dim,threads,max_depth,a,b)

        print("finished test, result - \n")
        print(dequantize_to_float32(result))
        print("baseline result - \n")
        #print(baseline_res.astype(np.int))
        #print(highlight_differences(result,baseline_res))
        #print((abs(result-baseline_res)).astype(np.uint))
        print("stats_total_cycles            :  " +str(stats_total_cycles  ))
        print("stats_total_thread_mult_ops   :  " +str(stats_ops_total ))
        print("stats_zero_ops %              :  " +str(100*stats_zero_ops/stats_ops_total))
        print("stats_1thread_mult_ops %      :  " +str(100*stats_1thread_mult_ops/stats_ops_total))
        print("stats_multi_thread_mult_ops % :  " +str(100*threads*stats_multi_thread_mult_ops/stats_ops_total ))
        print("stats_alu_not_utilized %      :  " +str(100*stats_alu_not_utilized/stats_alu_total ))
        print("stats_buffer_fullness_acc     :  " +str(stats_buffer_fullness_acc  ))
        print("stats_buffer_max_fullness     :  " +str(stats_buffer_max_fullness  ))
        


        
    """def test_example2(self):
        
        np.random.seed(0)
        dim = 14
        threads = 2
        alu_num = 1
        max_depth = 64
        push_back = True
        low_prec_mult = True
        
        a_w = 40
        a_h = 40
        a_c = 40
        zero_per = 0.7
        b_w = a_c
        b_h = 40
        a = np.random.randint(0,5,size = (a_w,a_h,a_c))
        b = np.random.randint(0,5,size = (b_w,b_h))

        a_num_zeros = int(zero_per * a_w * a_h * a_c)
        a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
        a.ravel()[a_zero_indices] = 0
        b_num_zeros = int(zero_per * b_w * b_h )
        b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
        b.ravel()[b_zero_indices] = 0

        best_dim = 10
        best_dim_time = 10

        for dim in range(10,20):
            start = time.time()
            result_tuple = m.run_uint8(dim,threads,alu_num,max_depth,a,b,push_back,low_prec_mult,False)
            end = time.time()
            cur_time = end - start
            if cur_time < best_dim_time:
                best_dim = dim
                best_dim_time = cur_time
        print("best dim: "+str(best_dim))
        print("best time: "+str(best_dim_time))"""
            


        

    #def test_vectorize(self):
    #    x1 = np.array([[0, 1], [2, 3]])
    #    x2 = np.array([0, 1])
    #    res = np.array([[ 1.               ,  1.381773290676036],
    #                    [ 1.909297426825682,  0.681422313928007]])
    #    y = m.vectorize_example1(x1, x2)
    #    np.testing.assert_allclose(y, res, 1e-12)

    #def test_readme_example1(self):
    #    v = np.arange(15).reshape(3, 5)
    #    y = m.readme_example1(v)
    #    np.testing.assert_allclose(y, 1.2853996391883833, 1e-12)

