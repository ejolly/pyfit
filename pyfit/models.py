from __future__ import division, print_function

'''
pyfit model classes
=======================
Main model class

'''

__all__ = ['CompModel']
__author__ = ['Eshin Jolly']
__license__ = 'MIT'

from lmfit import Parameters, minimize, fit_report
from operator import attrgetter
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

class CompModel(object):
    """
    Base computational model class.

    This is the basic class that represents model and data information together in a single object. It has methods to fit an objective function (func), with a design matrix (X) to some data (Y). It uses the minimize object from the lmit package, itself a wrapper around scipy.optimize to perform all operations. Objective functions should always return a vector of residuals (predictions - Y).

    Only the 'least_squares' methods can utilize different loss functions. All other optimizers will use SSE by default.

    Args:
        func: objective function to minimize
        data: pandas dataframe of all independent and dependent variables
        outcome_var: name of the column in data that refers the dependent variable to be modeled
        params_to_fit: dict of {'param_1':[init_val],'param_2':[init_val]}, OR {'param_1':[lb,ub],'param_2':[lb_ub]}
        group_var: name of the column in data that refers to a grouping variable to fit separate models to each sub-group within data
        loss: the type of loss to use to minimize the objective function; Default is sse; Must be one of: 'sse' or 'linear', 'll', 'soft_l1', 'huber', 'cauchy', or 'arctan'; default is sse/linear
        algorithm: what optimization algorithm to use; Default is least_squares; Must be one of: 'leastsq','least_squares','differential_evolution','brute','nelder','lbfgsb','powell','cg','newton','cobyla','tnc','trust-ncg','dogleg','slsqp'
        n_starts: optional, number of random initializations with starting values; default (100)
        extra_args: optional additional keword arguments to the objective function


    """
    def __init__(
    self,
    func,
    data,
    outcome_var,
    params_to_fit,
    group_var = None,
    loss = 'sse',
    algorithm = 'least_squares',
    extra_args = None,
    n_starts = 100):

        assert isinstance(data,pd.DataFrame), "Data must be pandas dataframe"

        assert isinstance(outcome_var,str), "outcome_var must be a string referring to the column name of the value to predict in data"

        assert callable(func), 'Objective function must be a callable python function!'

        if group_var is not None:
            assert isinstance(group_var,str), "group_var must be a string reffering to a column name of the grouping variable in data"

        assert algorithm in ['lm','least_squares','differential_evolution','brute','nelder','lbfgsb','powell','cg','newton','cobyla','tnc','trust-ncg','dogleg','slsqp'], 'Invalid algorithm, see docstring or lmfit/scipy docs for acceptable algorithms'

        assert loss in ['sse','linear', 'll', 'soft_l1', 'huber', 'cauchy', 'arctan'], 'Invalid loss, see docstring for acceptable losses'

        self.func = func
        if loss == 'll':
            raise NotImplementedError("-log liklihood is not yet implemented!")
        self.loss = loss
        self.algorithm = algorithm
        self.data = data
        self.outcome_var = outcome_var
        self.group_var = group_var
        self.params_to_fit = params_to_fit
        self.extra_args = extra_args
        self.n_starts = n_starts
        self.fitted_params = None
        self.preds = None
        self.MSE = None
        self.corr = None
        self.fitted = False

    def __repr__(self):

        if self.group_var:

            group_shapes = self.data.drop(self.outcome_var,axis=1).groupby(self.group_var).apply(lambda x: x.shape).unique()

            return '%s(X=%s, Y=%s, n_groups=%s, loss=%s, num_params=%s, fitted=%s)' % (
            self.__class__.__name__,
            group_shapes,
            self.outcome_var,
            self.data[self.group_var].nunique(),
            self.loss,
            len(self.params_to_fit.keys()),
            self.fitted,
            )
        else:
            return '%s(X=%s, Y=%s, n_groups=%s, loss=%s, num_params=%s, fitted=%s)' % (
            self.__class__.__name__,
            self.data.drop(self.outcome_var,axis=1).shape,
            self.outcome_var,
            1,
            self.loss,
            len(self.params_to_fit.keys()),
            self.fitted,
            )

    def fit(self,**kwargs):
        """
        Fit objective function by iterated random starts. Will uniformally sample within bounds if bounds are provided, otherwise will uniformally sample within a window +/- 5 (default) of initial parameters.

        Args:
            search_space: optional, window extent of uniform search for unbounded parameters (+/- iniital value)
            nan_policy: optional, how to handle nans; 'raise'-raise an error, 'propagate' (default)-don't update on that iteration but continue fitting, 'omit'-ignore non-finite values
            corr_type: optional, what type of correlation to use to assess post-fit (not used during fitting); pearson (default) or spearman
            kwargs: additional arguments to minimize() from lmfit

        """

        if self.params_to_fit is None and 'parameters' not in kwargs:
            raise IOError("Parameter(s) information is missing!")

        nan_policy = kwargs.pop('nan_policy','propagate')
        corr_type = kwargs.pop('corr_type','pearson')

        #How far around init unbounded params to grab random inits from
        search_space = kwargs.pop('search_space',[5])
        if isinstance(search_space,float) or isinstance(search_space,int):
            search_space = [search_space]

        #Loop over random initializations
        fitted_models = []
        for i in range(self.n_starts):

            #Make parameters
            params = self._make_params(search_space)

            #Make function call dict
            call = {}

            call['nan_policy'] = nan_policy
            call['fcn'] = self.func
            call['params'] = params

            #The default name Levenberg-Marquardt (leastsq) is confusing so have the user provide 'lm' instead and translate it here
            if self.algorithm == 'lm':
                call['method'] = 'leastsq'
            else:
                call['method'] = self.algorithm

            call['args'] = [self.data]
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
            else:
                warnings.warn("Model did not converge...")

        if fitted_models:
            #Make accessible some stats about the best fitting model
            self.best_fit = min(fitted_models, key=attrgetter('chisqr'))
            self.fitted_params = dict(self.best_fit.params.valuesdict())
            self.preds = self.best_fit.residual + self.data[self.outcome_var]
            if corr_type == 'pearson':
                corrs = pearsonr(self.preds,self.data[self.outcome_var])
            elif corr_type == 'spearman':
                corrs = spearmanr(self.preds,self.data[self.outcome_var])
            self.corr = {'r':corrs[0],'p':corrs[1]}
            self.MSE = np.mean(self.best_fit.residual**2)
            self.fitted = True
        else:
            warnings.warn("Fit failure no parameters found")
            #Return nans but with same attributes as a fit model
            self.fitted_params = self.params_to_fit.copy()
            for k,v in self.fitted_params.iteritems():
                self.fitted_params[k] = np.nan
            self.corr = {'r':np.nan,'p':np.nan}
            self.best_fit = dict(self.fitted_params)
            self.best_fit['chisqr'] = np.nan
            self.best_fit['redchi'] = np.nan
            self.best_fit['aic'] = np.nan
            self.best_fit['bic'] = np.nan
            self.corr['r'] = np.nan
            self.corr['p'] = np.nan
            self.best_fit['MSE'] = np.nan
            self.preds = np.nan
            self.MSE = np.nan
            self.fitted = 'Unsuccessful'

    def fit_group(self,group_name=None,verbose=1,**kwargs):
        """
        Fit a model to each member of 'group_name'.

        Args:
            group_name: str, must be a column name that exists in data
            verbose (int): 0- no printed output; 1- print fit failures only (default); 2- print fit message for each group

        """

        if group_name is None:
            group_name = self.group_var

        assert group_name is not None, "Grouping variable not set!"
        assert group_name in self.data.columns, "Grouping variable not found in data!"

        out = pd.DataFrame(columns=[self.group_var]+ self.params_to_fit.keys()+['chi-square','reduced_chi-square','AIC','BIC','corr_r','corr_p','MSE','fitted'])

        if verbose == 2:
            print("Fitting model to %s groups..." % self.data[group_name].nunique())

        for i,group in enumerate(self.data[group_name].unique()):
            if verbose == 2:
                print('Fitting group %s' % group)
            group_model = CompModel(
            func = self.func,
            data = self.data.loc[self.data[group_name] == group,:].reset_index(drop=True),
            outcome_var = self.outcome_var,
            loss = self.loss,
            algorithm = self.algorithm,
            params_to_fit = self.params_to_fit,
            extra_args = self.extra_args,
            n_starts = self.n_starts
            )
            group_model.fit(**kwargs)
            if group_model.fitted == 'Unsuccessful':
                if verbose == 1:
                    print("Group {} failed to fit".format(group))
                out_dat = dict(group_model.fitted_params)
                out_dat['chi-square'] = group_model.best_fit['chisqr']
                out_dat['reduced_chi-square'] = group_model.best_fit['redchi']
                out_dat['AIC'] = group_model.best_fit['aic']
                out_dat['BIC'] = group_model.best_fit['bic']
                out_dat['corr_r'] = group_model.corr['r']
                out_dat['corr_p'] = group_model.corr['p']
                out_dat['MSE'] = group_model.MSE
                out_dat['fitted'] = group_model.fitted
                out_dat[self.group_var] = group
            else:
                out_dat = dict(group_model.fitted_params)
                out_dat['chi-square'] = group_model.best_fit.chisqr
                out_dat['reduced_chi-square'] = group_model.best_fit.redchi
                out_dat['AIC'] = group_model.best_fit.aic
                out_dat['BIC'] = group_model.best_fit.bic
                out_dat['corr_r'] = group_model.corr['r']
                out_dat['corr_p'] = group_model.corr['p']
                out_dat['MSE'] = group_model.MSE
                out_dat['fitted'] = group_model.fitted
                out_dat[self.group_var] = group
            out = out.append(out_dat,ignore_index=True)
            del group_model
        self.fitted = True
        self.group_fits = out
        if verbose > 0:
            print("Fitting complete!")

    def summary(self):
        """
        Summarize fit information.
        """

        assert self.fitted, "Model has not been fit yet!"

        if self.group_var:

            #Compute average fit stats
            summary = self.group_fits.drop(self.group_var,axis=1).agg({'mean','std'})

            if self.algorithm == 'least_squares':
                print(
                """[[Group Mean Summary]]\n
                Num groups = %s\n
                Algorithm  = %s (TRF)\n
                Loss       = %s\n
                Successes  = %s\n
                """
                % (self.data[self.group_var].nunique(),self.algorithm,self.loss,self.group_fits['fitted'].value_counts()[True]))
            else:
                print(
                """[[Group Mean Summary]]\n
                    Num groups          = %s\n
                    Algorithm           = %s\n
                    Loss                = %s\n
                    Successes           = %s\n
                """
                % (self.data[self.group_var].nunique(),self.algorithm,self.loss,self.group_fits['fitted'].value_counts()[True]))

            #Display summary
            return summary

        else:
            if self.algorithm == 'least_squares':
                diag_string = '\n[[Diagnostics]] \n    Algorithm: %s (TRF) \n    Loss: %s \n    Success: %s' % (self.best_fit.method, self.loss, self.best_fit.success)

            else:
                diag_string = '\n[[Diagnostics]] \n    Algorithm: %s \n    Loss: %s \n    Success: %s' % (self.best_fit.method, self.loss, self.best_fit.success)
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
                if len(search_space) == 1:
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

    def _check_algo(self):

        """
        Raise a warning if the user provided bounds on parameters to fit, but requested an algorithm that doesn't support bounded optimization.
        """

        if np.sum([len(elem)>1 for elem in self.params_to_fit.values()]):
            if self.algorithm in ['lm','brute','nelder','powell','cg','newton','trust-ncg','dogleg']:
                warnings.warn("Requested algorithm does not support bounded optimization. Bounds will be ignored!")

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
