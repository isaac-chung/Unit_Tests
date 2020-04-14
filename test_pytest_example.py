import pytest
import numpy as np

from testfuns import rosen, rastrigin
from testfuns import rosen_der
from testfuns import relativeError
from testfuns import LinearModel
from testfuns import gradientDescent
           

PRECISION = 3
    
"""
Testing if the test functions are implemented correctedly.
"""

@pytest.mark.testfun
@pytest.fixture
def setdim():
    n = 1000
    d = 20
    return (n,d)

@pytest.mark.testfun
def test_rosen(setdim):
    n,d = setdim
    vec = rosen(np.ones(setdim))
    assert vec.shape[0] == 1000
    assert len(vec.shape) == 1
    assert vec.all() == 0.

@pytest.mark.testfun
def test_rastrigin(setdim):
    vec = rastrigin(np.zeros(setdim))
    assert vec.shape[0] == 1000
    assert len(vec.shape) == 1
    assert vec.all() == 0.
        
        
"""
Loss functions
"""
@pytest.fixture
def setPredTest():
    y_test = np.array([1.,1.,1.])
    y_pred = np.array([2.,1.,0.])
    return (y_test, y_pred)

def test_relativeError(setPredTest):
    y_test, y_pred = setPredTest
    err = relativeError(y_test, y_pred)
    assert err == np.sqrt(2)/np.sqrt(3)
        
        
"""
Testing LinearModel class functions
"""
@pytest.mark.linearmodel
@pytest.fixture
def LM_init():
    X = np.tile(0.1*np.arange(5)+0.1,(3,1)).T
    w = np.array([2,0.5,1.])
    y = np.array([0.35,0.7,1.05,1.4,1.75])
    return (X,y,w)

@pytest.mark.linearmodel
def testInit():
    lm = LinearModel()
    assert lm.w == None
    assert lm.d == None

@pytest.mark.linearmodel
def testFit(LM_init):
    X,y,w = LM_init
    trainedW = LinearModel(w0=w+0.1).fit(X,y)
    np.testing.assert_array_almost_equal(trainedW,w)

@pytest.mark.linearmodel
def testPredict(LM_init):
    X,y,w = LM_init
    lm = LinearModel(w=w)
    pred = lm.predict(X)
    assert pred.shape == y.shape
    np.testing.assert_array_almost_equal(pred,y)
        
        
"""
Testing optimizers
"""
@pytest.mark.optimizers
@pytest.fixture
def setopt():
    d = 20
    X = np.ones((d,))
    return (X,d)

@pytest.mark.optimizers
def testRosenDer(setopt):
    X,d = setopt
    rg = rosen_der(X)
    assert rg.shape == (d,)

@pytest.mark.optimizers
def testGD(setopt):
    X,d = setopt
    x0 = np.ones((d,))+0.1
    res = gradientDescent(x0,rosen_der)
    assert res.shape == x0.shape
    np.testing.assert_array_almost_equal(res,X,PRECISION)
    
