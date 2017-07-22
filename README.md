# pyfit

Computational model fitting toolbox in python.

#### [Description](#about)  
#### [Example Usage](#example-usage)  
#### [Optimizer Reference](#opt-ref)  

## Description <a name="about"></a>  
This toolbox is designed to be a *high-level* pythonic alternative to something like Matlab's fmincon or fminunc, which require minimal user input and generally solve a large class of problems well. Additionally it has some convenience features like fitting multiple identical models in quick succession (e.g. modeling each participant in an experiment), plotting, and predicting new data with a previously fit model.

This toolbox uses some of the core functionality of the great [lmfit](https://github.com/lmfit/lmfit-py) package which itself wraps [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html.). While access to highly customizable optimization is mostly possible by passing in the relevant scipy optimizer arguments and settings, it might be more beneficial to use either of those two packages if complete customization is what you're after.

## Example usage <a name="example-usage"></a>
```
from pyfit.models import CompModel

#Define an exponential model that we'll use as our objective function to minimize
#Our model always expects this objective function to return an array of residual values

def demand_curve(params_to_estimate, X, Y, endowment_range=1.25):

    "Demand curve objective function. Takes X vector as input, has 2 free parameters we're trying to solve for, and 1 additional fixed parameter.""

    ### PARAMETERS TO ESTIMATE ###
    Q0 = params_to_estimate['Q0']
    alpha = params_to_estimate['alpha']

    ### FIXED PARAMETERS/INPUTS ###
    C = X
    k = endowment_range

    ### MODEL ###
    predicted_y = np.log( Q0 ) + k * (np.exp( -1 * alpha * Q0 * C) - 1)
    residuals = predicted_y - Y

    return residuals

#Setup our optimizer object
from pyfit.models import CompModel

#Try it with the trf least squares algorithm and a l1 loss first

lstsq = CompModel(
    demand_curve,X,Y,algorithm='least_squares',
    loss='soft_l1',
    params_to_fit = {'Q0':[0,5],'alpha':[.1]},
    extra_args = {'endowment_range':1.25})

#For the 'Q0' parameter our optimizer is going to constrain its search to the interval [0,5], randomly initializing within that range multiple times. For the 'alpha' parameter, our optimizer is not going to constrain its search, but will randomly initializing within a range of values centered on our initial search value [0.1]

lstsq.fit()

#Get output

lstsq.summary()

#Now lets try it using Nelder-Mead which is an unconstrained optimizer

nelder = CompModel(
    demand_curve,X,Y,algorithm='nelder',
    params_to_fit = {'Q0':[.1],'alpha':[.1]},
    extra_args = {'endowment_range':1.25})

#Though we're not providing bounds for constrained search, let's constrain the window size for random initialize to (init_val - .1) - (init_val + 10); this window does not have to be symmetrical about the initial value

nelder.fit(search_space= [.1,10])
nelder.summary()
```

## Optimizer Reference <a name="opt-ref"></a>
*A full list of scipy optimizers is [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize), while least squares specific algorithms are [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)*

While picking the "right" algorithm depends on the problem you're trying to solve and the properties of your objective function, the following are fairly robust for a large class of problems and a reasonable first bet, and none of them require you to compute the gradient ahead of time or even know what it is (though that can help). A great comprehensive overview of the scope of these issues is [here](http://www.scipy-lectures.org/advanced/mathematical_optimization/).

A reasonable optimizer-to-try trajectory would be:  
**Unbounded and Unconstrained**: `lm -> nelder`  
**Bounded and Unconstrained**: `least_squares -> lbfgsb`  
**Bounded and Constrained**: `slsqp or cobyla`  


#### Nelder-Mead
- Good for well-conditioned, high-dimensional problems, and noisy measurements Nelder-Mead (`method = 'nelder'`) as it doesn't require having or computing function gradients, just the function evaluations themselves
    - *Cannot handle bounds or constraints*

#### L-BFGS-B
- Limited memory version of the BFGS algorithm (`method = lbfgsb`) which is a gradient (derivative) based method that's flexible enough to handle a large class of problems, at the cost of potentially being slower or slightly less accurate than simplex methods on well conditioned problems
    - *Can handle bounds*
    - *Computing gradients ahead of time can improve*

#### SLSQP
- Sequential Least Squares Programming algorithm (`method = slsqp`) which particularly useful for objective functions that are subject to both bounds and equality and/or inequality constraints
    - *Can handle bounds and constraints*
    - *Computing gradients ahead of time can improve*

#### Least Squares - Trust Region Reflective
- Least squares based algorithm (`method = 'least_squares'`) thats good for large problems that that do or don't require bounds. Without bounds it performs similarly to the efficient Levenberg-Marquardt algorithm and is generally flexible and robust for a large class of problems
    - *Can handle bounds*
    - *Can handle several costs (e.g. arctan, L1, etc)*

#### Levenberg-Marquardt
- The optimizer in scipy.optimize.lstsq (`method = 'lm'`), a generally fast and robust algorithm thats a good first go-to for unconstrained, unbounded problems
    - *Cannot handle bounds*


### Todo
- Raise warning if provided algorithm that ignores bounds, or visa versa, because lmfit auto falls back to leastsq or nelder.

- Make search_window flexible to take values separately for each parameter
- Plotting
    - Plot generate model curve
    - Plot correlation between predicted values and real values
