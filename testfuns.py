import autograd.numpy as np
from autograd import grad
from sklearn.base import BaseEstimator

def rastrigin(x):
    """ (N,D) -> (N,) Returns the rastrigin test function for a given x
    """
    D = x.shape[1]
    rast_vec = np.power(x, 2) - 10*np.cos(2*np.pi*x)
    return 10*D + rast_vec.sum(axis=1)


def rosen(x):
    """ (N,D) -> (N,) Returns the rosenbrock test function for a given x
    """
    rosen_vec = 100.0*(x[:,1:]-x[:,:-1]**2)**2 + (1-x[:,:1])**2
    return rosen_vec.sum(axis=1)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def relativeError(y_test,y_pred):
    """ (N,), (N,) -> (N,) Returns the relative error between the target and prediction vectors
    """
    assert y_test.shape == y_pred.shape
    return np.sqrt(np.sum(np.square(y_pred-y_test)))/np.sqrt(np.sum(np.square(y_test)))


def mse(y_test,y_pred):
    """ (N,), (N,) -> (N,) Returns the mean squared error between the target and prediction vectors
    """
    assert y_test.shape == y_pred.shape
    return 0.5*np.sum((y_pred-y_test)**2)/y_test.shape[0]


def gradientDescent(x, gradfun, args=(), a=0.001, n=10000):
    """
    Full batch gradient descent
    """
    g = gradfun(x, *args)
    assert g.shape == x.shape, g.shape
    for i in range(n):
        g = gradfun(x, *args)
        x = x - a*g
    return x


class LinearModel(BaseEstimator):
    def __init__(self, w=None, w0=None):
        self.w = w
        self.w0 = w0
        self.d = None
    
    def fit(self,X,y):
        assert y.shape[0] == X.shape[0]
        if self.w0 is None:
            self.w0 = np.random.uniform(0,2,(X.shape[1],))
        self.w = gradientDescent(self.w0, grad(self.loss,0), args=(X,y),a=0.1,n=1000)
        return self.w
        
    def predict(self, X):
        assert (self.w is None) == False
        if self.d is None:
            self.d = self.w.shape[0]
        return X.dot(self.w)
    
    def loss(self,w,X,y):
        return mse(y,np.dot(X,w))
        
        
        