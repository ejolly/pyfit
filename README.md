# pyfit

Computational model fitting in python.  

This toolbox is designed to be a *high-level* pythonic alternative to something like Matlab's fmincon or fminunc, which require minimal user input and generally solve a large class of problems well. However, it has numerous enhancements and the major difference between this toolbox and existing tools are: tight integration with pandas, multiple random initializations of optimizers, and the ability to simply fit separate models to grouped data (e.g. individual participants in an experiment). 

#### [Description](#about)  
#### [Defining an objective function](#def-obj)  
#### [Example Usage](#example-usage)  
#### [Optimizer Reference](#opt-ref)  

## Description <a name="about"></a>  
At it's core this toolbox requires a user to have some data to model, and an objective function to try to fit to the data via optimization routines built into scipy.  

This toolbox uses some of the core functionality of the great [lmfit](https://github.com/lmfit/lmfit-py) package which itself wraps [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html.). While access to highly customizable optimization is mostly possible by passing in the relevant scipy optimizer arguments and settings, it might be more beneficial to use either of those two packages if complete customization is what you're after.

## Defining an objective function <a name="def-obj"></a>  
The underlying routines expect that an objective function adheres to a particular type of definition structure. This requires 4 key pieces:  
1. The first argument should be a dictionary object with named parameters to estimate
2. The second argument should be a pandas dataframe that contains the columns of predictor and outcome variables to use during modeling
3. Any additional arguments used by the objective function
4. The function *must* return residuals (predictions - outcome var)  

Here is the simplest example:  
```
def sample_obj(params_to_estimate, df, fixed_var):
    '''
    Fit a model where:
    Y = (X*A + Z)**2./Z

    X: independent variable
    A: free parameterize to estimate
    Z: fixed parameter
    '''

    #Unpack free parameter based on its name
    A = params_to_estimate['param_1']

    #Model
    prediction = (df['predictor_var'] * A + fixed_var)**2./fixed_var

    #Return residual
    return prediction - df['outcome_var']
```

## Example usage <a name="example-usage"></a>

**Simple single constrained optimization**  

Now that an objective function is defined we can setup our model optimizer object. Here's the simplest example which will do least-squares optimization by default, searching within the specified bounds [0,5] and randomly initializing the search multiple times.


```
from pyfit.models import CompModel

lstsq = CompModel(
    func = sample_obj,
    data = df,
    Y_var = 'outcome_var',
    params_to_fit = {'param_1':[0,5]},
    extra_args = {'fixed_var':1.25})

#Fit it
lstsq.fit()

#Print summary
lstsq.summary()
```

**Group unconstrained optimization**  

Lets try a different optimizer without bounds. We can also fit multiple models simultaneously if we have a grouping indicator in our dataframe. In this example let's fit a model to each participant in an experiment. Because we haven't specified bounds, the optimizer will instead run many random initializations in a uniform symmetric window around the initial value of our parameter.  

```
nelder = CompModel(
    func = sample_obj,
    data = df,
    Y_var = 'outcome_var',
    group_var = 'Subject',
    algorithm='nelder',
    params_to_fit = {'param_1':[2]},
    extra_args = {'fixed_var':1.25})

#Fit it
nelder.fit()

#Print group summary, averaging over fit statistics and parameter values
nelder.summary()
```

**Multi-parameter mixed optimization**  

Here we're going to fit a model with an objective function that has two free parameters 'Q0' and 'alpha', and do constrained optimization for only one of them. For the 'Q0' parameter our optimizer is going to constrain its search to the interval [0,5], randomly initializing within that range multiple times. For the 'alpha' parameter, our optimizer is not going to constrain its search, but will randomly initializing within a range of values centered on our initial search value [0.1]

```
#Using L-BFGS-B optimizer which can handle bounds

model = CompModel(
    func = two_param_obj,
    data = df,
    Y_var = 'outcome_var',
    algorithm='lbfgsb',
    params_to_fit = {'Q0':[0,5],'alpha':[.1]})

#We can even control the size and shape of the uniform sampling window around our 'alpha' parameter. Let's define the window size as (init_val - .1) - (init_val + 10); this window does not have to be symmetrical about the initial value

#Fit with specified search window
model.fit(search_space= [.1,10])

#Print summary
model.summary()
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
