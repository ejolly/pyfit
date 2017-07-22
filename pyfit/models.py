from __future__ import division, print_function

'''
pyfit model classes
=======================
Main model class

'''

__all__ = ['CompModel']
__author__ = ['Eshin Jolly']
__license__ = 'MIT'

from collections import OrderedDict
from lmfit import Parameters, minimize, fit_report
from operator import attrgetter
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class CompModel(object):
    """
    Base computational model class.

    This is the basic class that represents model and data information together in a single object. It has methods to fit an objective function (func), with a design matrix (X) to some data (Y). It uses the minimize object from the lmit package, itself a wrapper around scipy.optimize to perform all operations. Objective functions should always return a vector of residuals (predictions - Y).

    Only the 'least_squares' methods can utilize different loss functions. All other optimizers will use SSE by default.

    Args:
        func: objective function to minimize
        X: numpy array or pandas dataframe of design matrix/independent  variables
        Y: data values/dependent variables to fit to
        loss: the type of loss to use to minimize the objective function must be one of: sse, ll, soft_l1, huber, cauchy, or arctan; default is sse
        algorithm: what optimization algorithm to use must be one of: 'leastsq','least_squares','differential_evolution','brute','nelder','lbfgsb','powell','cg','newton','cobyla','tnc','trust-ncg','dogleg','slsqp'
        params_to_fit: dict of {'param_1':[init_val],'param_2':[init_val]}, OR {'param_1':[lb,ub],'param_2':[lb_ub]}
        nrep: optional, number of random initializations with starting values
        extra_args: optional additional keword arguments to the objective function


    """
    def __init__(
    self,
    func,
    X = None,
    Y = None,
    loss = 'sse',
    algorithm = 'trf',
    params_to_fit = None,
    extra_args = None,
    nrep = 1000):

        if X is not None:
            assert (isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)), "Design matrix must be pandas dataframe or numpy array"
        if Y is not None:
            assert (isinstance(Y,pd.DataFrame) or isinstance(Y,pd.Series) or isinstance(Y,np.ndarray)), "Y vector must be a pandas data type or numpy array"
        if func is not None:
            assert callable(func), 'Objective function must be a callable python function!'

        assert algorithm in ['trf', 'dogbox','lm','least_squares','differential_evolution','brute','nelder','lbfgsb','powell','cg','newton','cobyla','tnc','trust-ncg','dogleg','slsqp'], 'Invalid algorithm, see docstring or lmfit/scipy docs for acceptable algorithms'

        assert loss in ['sse', 'linear','ll', 'soft_l1', 'huber', 'cauchy', 'arctan'], 'Invalid loss, see docstring for acceptable losses'

        self.func = func
        if loss == 'll':
            raise NotImplementedError("-log liklihood is not yet implemented!")
        self.loss = loss
        self.algorithm = algorithm
        self.X = X
        self.Y = Y
        self.params_to_fit = params_to_fit
        self.extra_args = extra_args
        self.nrep = nrep
        self.fitted_params = None
        self.preds = None
        self.MSE = None
        self.corr = None
        self.fitted = False

    def __repr__(self):
        return '%s(X=%s, Y=%s, loss=%s, num_params=%s, fitted=%s)' % (
        self.__class__.__name__,
        self.X.shape,
        self.Y.shape,
        self.loss,
        len(self.params_to_fit.keys()),
        self.fitted,
        )

    def fit(self,**kwargs):
        """
        Fit objective function by iterated random starts. Will uniformally sample within bounds if bounds are provided, otherwise will uniformally sample within a window +/- 5 (default) of initial parameters.

        Args:
            search_space: optional, window extent of uniform search for unbounded parameters (+/- iniital value)
            kwargs: additional arguments to minimize() from lmfit

        """

        if self.params_to_fit is None and 'parameters' not in kwargs:
            raise IOError("Parameter(s) information is missing!")

        nan_policy = kwargs.pop('nan_policy','propagate')

        #How far around init unbounded params to grab random inits from
        search_space = kwargs.pop('search_space',5)

        #Loop over random initializations
        fitted_models = []
        for i in xrange(self.nrep):

            #Make parameters
            params = self._make_params(search_space)

            #Make function call dict
            call = {}

            call['nan_policy'] = nan_policy
            call['fcn'] = self.func
            call['params'] = params
            call['method'] = self.algorithm
            call['args'] = (self.X,self.Y)
            if self.extra_args is not None:
                call['kws'] = self.extra_args
            #Other loss functions only work for least_squares
            if self.algorithm == 'least_squares':
                if self.loss == 'sse':
                    call['loss'] = 'linear'
                else:
                    call['loss'] = self.loss

            #Fit
            call.update(kwargs) #additional kwargs
            fit = minimize(**call)

            if fit.success:
                if fitted_models:
                    if fit.chisqr < fitted_models[-1].chisqr:
                        fitted_models.append(fit)
                else:
                    fitted_models.append(fit)

        if not fitted_models:
            warnings.warn("No successful model fits!!!!")
        else:
            #Get the best model
            self.best_fit = min(fitted_models, key=attrgetter('chisqr'))
            self.fitted_params = self.best_fit.params.valuesdict()
            self.preds = self.best_fit.residual + self.Y
            corrs = pearsonr(self.preds,self.Y)
            self.corr = OrderedDict({'r':corrs[0],'p':corrs[1]})
            self.MSE = np.mean(self.best_fit.residual**2)
            self.fitted = True


    def summary(self):
        """
        Summarize fit information.
        """

        assert self.fitted, "Model has not been fit yet!"

        if self.algorithm == 'least_squares':
            diag_string = '\n[[Diagnostics]] \n    Algorithm: %s (TRF) \n    Loss: %s \n    Success: %s' % (self.best_fit.method, self.loss, self.best_fit.success)

        else:
            diag_string = '\n[[Diagnostics]] \n    Method: %s \n    Success: %s' % (self.best_fit.method, self.best_fit.success)

        print(
        fit_report(self.best_fit) + diag_string

        )

    def _make_params(self,search_space):
        """
        Make the parameter grid. Default to search in a space of +/- 5 around unbounded parameters.

        """

        params = Parameters()
        for k,v in self.params_to_fit.iteritems():
            if len(v) == 1:
                if len(search_space == 1):
                    val = np.random.uniform(v[0]-search_space[0],v[0]+search_space[0])
                else:
                    val = np.random.uniform(v[0]-search_space[0],v[0]+search_space[1])
                params.add(k,value=val)
            elif len(v) == 2:
                val = np.random.uniform(v[0],v[1])
                params.add(k,value=val,min=v[0],max=v[1])
            else:
                raise ValueError("Parameters are not properly specified, should be a {'param':[init_val]} or {'param':[lb, ub]}")
        return params


    def predict(self):
        """Make predictions using fitted model parameters. Also computers MSE and correlation
        between predictions and Y vector."""

        raise NotImplementedError("Prediction not yet implemented.")

        #

        # assert self.fittedParams is not None, "No model parameters have been estimated! \nCall fit(), before predict()."
        #
        # self.preds = self.objFun(self.fittedParams,self.X)
        # self.MSE = np.mean(np.power(self.preds-self.Y,2))
        # self.predCorr = np.corrcoef(self.preds,self.Y)[0,1]
        # print("Prediction correlations: %s" % self.predCorr)
        return
