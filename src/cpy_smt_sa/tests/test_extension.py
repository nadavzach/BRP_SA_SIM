import cpy_smt_sa as m
from unittest import TestCase
import numpy as np
import random


class ExampleTest(TestCase):

   
    def test_example1(self):
        stats_zero_ops = 0
        stats_1thread_mult_ops = 0
        stats_multi_thread_mult_ops = 0
        stats_alu_not_utilized = 0
        stats_buffer_fullness_acc = 0
        stats_buffer_max_fullness = 0
        
        np.random.seed(0)
        dim = 3
        threads = 2
        alu_num = 1
        max_depth = 10 
        
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
        print("a array: \n")
        print(a)
        print("b array: \n")
        print(b)
        print("running first test with run_int8:")
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

