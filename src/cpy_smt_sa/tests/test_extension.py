import cpy_smt_sa as m
from unittest import TestCase
import numpy as np
import random


class ExampleTest(TestCase):

   
    def test_example1(self):
        dim = 3
        threads = 2
        alu_num = 1
        max_depth = 2 

        a_w = 5
        a_h = 5 
        a_d = 5 

        b_w = a_d 
        b_h = 5 
        a = np.ones((a_w,a_h,a_d))#np.random.randint(0,100,size = (a_w,a_h,a_d))
        b = np.ones((b_w,b_h ) )#np.random.randint(0,100,size = (b_w,b_h))
        print("a array: \n")
        print(a)
        print("b array: \n")
        print(b)
        print("running first test with run_int64:")
        result = m.run_int32(dim,threads,alu_num,max_depth,a,b)
        print("finished test, result - \n")
        print(result.astype(np.uint))

        

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

