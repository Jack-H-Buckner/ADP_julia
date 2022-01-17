"""
    BellmanOpperators

Provides methods to evaluate the Bellman equation for each of the POMDP objects. 

The primay function defined in this module is `bellman` which evaluates the bellman opperator give
a POMDP object, and initial state and a value function object. If an action variable `a` is provided 
then the algorithm will calcualte the expected value of taking that action in the current state. If
no action is provided then the function will solve for the best action using the `Optim.jl` package. 

There are two primary computational challenges related to solving the bellman opperator: optimization and 
integration. Seperate methos are defined to apply the most aproporate method depending on the input arguments.

For `POMDP_particleFilter` objects Montecarlo integration is used, and Gauss Hermite quadrature is used for 
`POMDP_kalmanFilter` objects. The implementation of the Gausian quadrature depends on weather or not the observation
model is linear. If it is linear then 

The optimizaiton method is chosen based on the Type of the POMDP.actions object. Nelder Mead is used for 
multivariate continuous aciton, Golden search for univariate continuous, and brute force is used for 
discrete action spaces. 
"""
module BellmanOpperators

#using Optim
#using Roots
include("MvGaussHermite.jl")
include("utils.jl")

"""
    propogate_observation_model(x_hat,Cov,H)

propogates state uncertianty through observation model if H is a function
the gassian quadrature is used if it is a matrix then a linear transform
is used. 

x_hat - mean estimate for x
Cov - covariance of beleif state
H - observation model, matrx of function 
"""
function propogate_observation_model(x_hat::AbstractVector{Float64},x_cov::AbstractMatrix{Float64},
                                     H::Function,Quad)
    MvGaussHermite.update!(Quad, x_cov)
    y_hat = MvGaussHermite.expected_value(H, Quad, x_hat)
    y_cov = utils.sum_mat(broadcast(v-> (H(v.+x_hat).-y_hat)*transpose(H(v.+x_hat).-y_hat), Quad.nodes).* Quad.weights)
    return y_hat, y_cov
end 


function propogate_observation_model(x_hat::AbstractVector{Float64},x_cov::AbstractMatrix{Float64},
                                     H::AbstractMatrix{Float64})
    y_hat = H * x_hat
    y_cov = H * x_cov * transpose(H)
    return y_hat, y_cov
end 



function integrate_bellman(x_hat, x_cov, POMDP_kalmanFilter)
end 

end # module 