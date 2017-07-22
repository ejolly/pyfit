# fitpy

Computational model fitting toolbox in python.

## Description


## Example Usage

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

### Todo
Raise warning if provided algorithm that ignores bounds, or visa versa, because lmfit auto falls back to leastsq or nelder.

Make search_window flexible to take values separately for each parameter
