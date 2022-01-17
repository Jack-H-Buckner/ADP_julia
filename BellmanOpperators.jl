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
using KalmanFilters
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


# struct POMDP_KalmanFilter{T} 
#     T!::Function # deterministic state transition function T(x,a) = x'
#     T::Function
#     actions::T
#     R::Function # reward function R(x,a,epsilon) = r (epsilon is a gausian random variable with vocariance Sigma_N)
#     H::Function # observaiton function 
#     Sigma_N::AbstractMatrix{Float64} # process noise covariance  
#     Sigma_O::AbstractMatrix{Float64} # observaiton noise covariance 
#     d_proc::Distribution{Multivariate,Continuous}
#     d_obs::Distribution{Multivariate,Continuous}
#     T_sim!::Function # simulates stochastic state transitions T(x,a) = x' + epsilon
#     G::Function # likelihood of observations yt - xt ~ N(0, Sigma_O) 
#     G_sim::Function # yt = xt + epsilon where epsilon ~ N(0, Sigma_O)
#     delta::Float64 # discount factor 
# end 

"""
    integrate_bellman(x_hat, x_cov, a, V, POMDP)

x_hat - mean state estimate
x_cov - mean covariance estimate
a - action taken
V - value function (takes state )
POMDP - 
"""
function integrate_bellman(x_hat::AbstractVector{Float64}, x_cov::AbstractMatrix{Flaot64}, 
                            a::AbstractVector{Float64}, V::Function, POMDP, Quad_x, Quad_y)
    
    ### time update 
    tu = KalmanFilters.time_update(x_hat,x_cov, x ->POMDP.T(x,a),  POMDP.Sigma_N)
    x_hat, x_cov = KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
    MvGaussHermite.update!(Quad_x,x_hat,x_cov)
    #observaton quadrature nodes 
    H = x -> POMDP.H(x,a) # define measurment function 
    y_hat, y_cov = propogate_observation_model(x_hat, x_cov,H,Quad_x)
    MvGaussHermite.update!(Quad_y,y_hat,y_cov)
    
    
    ### new states 
    new_states = broadcast(y -> KalmanFilters.measurement_update(x_hat, x_cov,y,
                    x ->  unknownGrowthRate.H(x,[0.1]),unknownGrowthRate.Sigma_O), Quad_y.nodes)
    
    
end 

end # module 