import unittest
import numpy as np

from testfuns import rosen, rastrigin
from testfuns import rosen_der
from testfuns import relativeError
from testfuns import LinearModel
from testfuns import gradientDescent
           

PRECISION = 3
    
class TestMinimum(unittest.TestCase):
    """
    Testing if the test functions are implemented correctedly.
    """
    def setUp(self):
        self.n = 1000
        self.d = 20
    
    def test_rosen(self):
        vec = rosen(np.ones((self.n,self.d)))
        self.assertEqual(vec.shape[0],self.n)
        self.assertEqual(len(vec.shape),1)
        self.assertAlmostEqual(vec.all(),0.)
        
    def test_rastrigin(self):
        vec = rastrigin(np.zeros((self.n,self.d)))
        self.assertEqual(vec.shape[0],self.n)
        self.assertEqual(len(vec.shape),1)
        self.assertAlmostEqual(vec.all(),0.)
        
        
class TestLossFunctions(unittest.TestCase):
    """
    Loss functions
    """
    def setUp(self):
        self.y_test = np.array([1.,1.,1.])
        self.y_pred = np.array([2.,1.,0.])
    
    def test_relativeError(self):
        err = relativeError(self.y_test, self.y_pred)
        self.assertAlmostEqual(err,np.sqrt(2)/np.sqrt(3))
        
        
class TestLinearModel(unittest.TestCase):
    """
    Testing LinearModel class functions
    """
    def setUp(self):
        self.X = np.tile(0.1*np.arange(5)+0.1,(3,1)).T
        self.w = np.array([2,0.5,1.])
        self.y = np.array([0.35,0.7,1.05,1.4,1.75])
        
    def testInit(self):
        lm = LinearModel()
        self.assertEqual(lm.w,None)
        self.assertEqual(lm.d,None)
        
    def testFit(self):
        trainedW = LinearModel(w0=self.w+0.1).fit(self.X,self.y)
        np.testing.assert_array_almost_equal(trainedW,self.w)
    
    def testPredict(self):
        lm = LinearModel(w=self.w)
        pred = lm.predict(self.X)
        self.assertEqual(pred.shape,self.y.shape)
        np.testing.assert_array_almost_equal(pred,self.y)
        
        
class TestOptimizer(unittest.TestCase):
    """
    Testing optimizers
    """
    def setUp(self):
        self.d = 20
        self.X = np.ones((self.d,))
        
    def testRosenDer(self):
        rg = rosen_der(self.X)
        self.assertTrue(rg.shape,(self.d,))
    
    def testGD(self):
        x0 = np.ones((self.d,))+0.1
        res = gradientDescent(x0,rosen_der)
        self.assertTrue(res.shape,x0.shape)
        np.testing.assert_array_almost_equal(res,self.X,PRECISION)
    
            
if __name__ == '__main__':
    unittest.main(verbosity=2)
    
    
    
    
    
    
