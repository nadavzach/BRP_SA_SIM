import cpy_smt_sa as m
import baseline_smt_sa as baseline
from unittest import TestCase
import numpy as np
import random

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
        max_depth = 10
        push_back = True
        
        a_w = 5
        a_h = 5 
        a_d = 5 
        
        b_w = a_d 
        b_h = 5 
        a = np.random.randint(0,255,size = (a_w,a_h,a_d))
        b = np.random.randint(0,255,size = (b_w,b_h))
        
        a_mask = np.random.randint(0,2,size = (a_w,a_h,a_d))
        b_mask = np.random.randint(0,2,size = (b_w,b_h))
        a=np.multiply(a,a_mask)
        b=np.multiply(b,b_mask)
        """print("a array: \n")
        print(a)
        print("b array: \n")
        print(b)
        print("running first test with run_int8:")"""
        result_tuple = m.run_uint8(dim,threads,alu_num,max_depth,a,b,push_back)
        result = result_tuple[0]
        stats_zero_ops              = result_tuple[1]
        stats_1thread_mult_ops      = result_tuple[2]
        stats_multi_thread_mult_ops = result_tuple[3]
        stats_buffer_fullness_acc   = result_tuple[4]
        stats_buffer_max_fullness   = result_tuple[5]
        stats_alu_not_utilized      = result_tuple[6]
        stats_total_cycles          = result_tuple[7]

        stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;

        baseline_res = baseline.run_uint8(dim,threads,max_depth,a,b)

        print("finished test, result - \n")
        print(result.astype(np.uint))
        print("baseline result - \n")
        print(baseline_res.astype(np.uint))
        highlight_differences(result,baseline_res)
        #print((result-baseline_res).astype(np.uint))
        print("stats_zero_ops %              :  " +str(100*stats_zero_ops/stats_ops_total             ))
        print("stats_1thread_mult_ops %      :  " +str(100*stats_1thread_mult_ops/stats_ops_total     ))
        print("stats_multi_thread_mult_ops % :  " +str(100*threads*stats_multi_thread_mult_ops/stats_ops_total ))
        print("stats_total_thread_mult_ops % :  " +str(stats_ops_total ))
        print("stats_alu_not_utilized        :  " +str(stats_alu_not_utilized     ))
        print("stats_buffer_fullness_acc     :  " +str(stats_buffer_fullness_acc  ))
        print("stats_buffer_max_fullness     :  " +str(stats_buffer_max_fullness  ))
        print("stats_total_cycles            : " +str(stats_total_cycles  ))


        
    #def test_example2(self):
    #    x = np.array([[0., 1.], [2., 3.]])
    #    res = np.array([[2., 3.], [4., 5.]])
    #    y = m.example2(x)
    #    np.testing.assert_allclose(y, res, 1e-12)

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

