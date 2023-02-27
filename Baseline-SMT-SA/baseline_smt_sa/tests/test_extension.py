import baseline_smt_sa as m
from unittest import TestCase
import numpy as np


class ExampleTest(TestCase):

    def test_example1(self):
    
        np.random.seed(0)
        dim = 3
        threads = 2
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
        result = m.run_uint8(dim,threads,max_depth,a,b)

        print("finished test, result - \n")
        print(result.astype(np.uint))


