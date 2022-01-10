"""
    BellmanOpperators

Provides methods to evaluate the Bellman equation for each of the POMDP objects. 

The primay function defined in this module is `Bellman` which evaluates the bellman opperator give
a POMDP object, and initial state and a value function object. If an action variable `a` is provided 
then the algorithm will calcualte the expected value of taking that action in the current state. If
no action is provided then the function will solve for the best action using the `Optim.jl` package. 

There are two primary computational challenges related to solving the bellman opperator: optimization and 
integration. Seperate methos are defined to apply the most aproporate method depending on the input arguments.

For `POMDP_particleFilter` objects Montecarlo integration is used, and Gause-Hermite quadrature is used for 
`POMDP_kalmanFilter` objects.

The optimizaiton method is chosen based on the Type of the POMDP.actions object. Nelder Mead is used for 
continuous aciton, and brute force is used for discrete action spaces. 
"""
module BellmanOpperators
using Optim
using 


end # module 