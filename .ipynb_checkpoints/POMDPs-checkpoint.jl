"""
Note: this is intended to have function to solve a range of problems, but I am planningto focus on the 
Simpelest (POMDPs) first and then fill in methods for the others over time. 

This module defines data structures to store the information required to define each of the
problem types treated by this library. these are

POMDP - partially observable markov decision process
MOMDP - mixed observability markov decision process
theta_POMDP - mixed observability markov decision process where all state varaible are directly observed
Hidden_MDP - a MDP (partialy observed or other wise) with one partially observed discrete state

In addition to these objects the module also defines and funciton `init` with methods that initailize
the relevant POMDP object given the data that was entered.

Dependencies:
`Distributions.jl` - in particualr multivariate normal 



Notes on action space:
- actions are seperated into two parts, actions for gathing informaiton -obs and actions for controling the system-a
- observations -obs only change the informaiton collection process while controls -a might effect onformaton and the systems dynamics
- The observaiton model is a function of both the observaiton specific action and the controls acitons
- the procss model only depends on the control actions. 
- this seperation makes computing the solution given perfect informaiton easier. 
"""
module POMDPs

using Distributions 
using IterTools


# struct boundedActions
#     n_dims::Int64
#     lower::AbstractVector{Float64} # leaving this abstract possible bottelneck?
#     upper::AbstractVector{Float64} # leaving this abstract possible bottelneck?
# end 

# struct continuousActions
#     n_dims::Int64
# end 

# struct discreteActions
#     n_actions::Int64
#     actions::AbstractVector{} # leaving this abstract possible bottelneck?
# end 


"""
Defines an arbitrary POMDP bsed on a transition
"""
struct POMDP_particleFilter
    T!::Function # stochastic state transition function:  T(x,a) = x'
    actions::AbstractVector{AbstractVector{Float64}}
    observations::AbstractVector{AbstractVector{Float64}}
    R::Function # (stochastic) Reward function: R(x,a) = r  
    G_sim::Function # stochatic observations function: G_sim(xt,a) = yt
    G::Function # likelihood function: G(y,x,a) = d/dy P(Y < y|x,a)
    delta::Float64 # discount factor 
end 

# function init(T!::Function, R::Function, G_sim::Function, G::Function, delta::Float64, 
#                 n_dims::Int64)
#     actions = continuousActions(n_dims)
#     return POMDP_particleFilter{continuousActions}(T!,actions, R, G_sim, G, delta)
# end 

# function init(T!::Function,R::Function, G_sim::Function, G::Function, delta::Float64,
#                 n_dims::Int64,upper::AbstractArray{Float64},lower::AbstractArray{Float64})
#     actions = boundedActions(n_dims, upper, lower)
#     return POMDP_particleFilter{boundedActions}(T!,actions, R, G_sim, G, delta)
# end 

function init(T!::Function,R::Function, G_sim::Function, G::Function, delta::Float64,
               actions,
               observations)#::AbstractVector{AbstractVector{Float64}}
    return POMDP_particleFilter(T!,actions,observations, R, G_sim, G, delta)
end 


"""
    POMDP_KalmanFilter

This object can be used to solve POMDPs with nonlinear process and observaitosn models
that have additive normally distributed process and observation noise with a constant covariance
given some aproporeate transformaiton. This object contains all of the informaiton required to 
used extended and unscented kalman filter methods to aproximate the beleif dynamics and particle
filter techniques. 

T! - xt+1 = T!(x+t,a) + epsilon_t deterministic state transition funciton
R - R(x,a) 
"""
struct POMDP_KalmanFilter
    T!::Function # deterministic state transition function T(x,a) = x'
    T::Function
    actions::AbstractVector{AbstractVector{Float64}}
    observations::AbstractVector{AbstractVector{Float64}}
    A::AbstractVector{Tuple{AbstractVector{Float64},AbstractVector{Float64}}}
    R::Function # reward function R(x,a) = r 
    R_obs::Function
    H::Function # observaiton function 
    Sigma_N::AbstractMatrix{Float64} # process noise covariance  
    Sigma_O::Function # maps actions to covarinace matrix observaiton noise covariance 
    d_proc::Distribution{Multivariate,Continuous}
    d_obs::Function
    T_sim!::Function # simulates stochastic state transitions T(x,a) = x' + epsilon
    G::Function # likelihood of observations yt - xt ~ N(0, Sigma_O) 
    G_sim::Function # yt = xt + epsilon where epsilon ~ N(0, Sigma_O)
    delta::Float64 # discount factor 
end 

function generateFunctions(T!::Function, T::Function, H::Function,
              Sigma_N::AbstractMatrix{Float64}, Sigma_O::Function,
            actions, observations )
    
    dx = size(Sigma_N)[1]
    dy = size(Sigma_O(actions[1], observations[1]))[1]

    d_proc = Distributions.MvNormal(zeros(dx),Sigma_N) 

    d_obs = (a,obs) -> Distributions.MvNormal(zeros(dy),Sigma_O(a, obs)) 
    
    function T_sim!(x,a)
        T!(x,a)
        x .+= reshape(rand(d_proc, 1),dx)
    end 
    
    G_sim = (x,a,obs) -> H(x,a, obs) .+ reshape(rand(d_obs(a,obs), 1),dy,1)
    G = (y,x,a,obs) -> pdf(d_obs(a,obs), y .- H(x,a,obs))
    return T_sim!, d_proc, d_obs, G_sim, G
end 

# function init(T!::Function, T::Function,R::Function, H::Function,
#               Sigma_N::AbstractMatrix{Float64}, Sigma_O::AbstractMatrix{Float64}, 
#               delta::Float64, n_dims::Int64)
    
#     T_sim!, d_proc, d_obs, G_sim, G = generateFunctions(T!::Function, T::Function, H::Function,
#               Sigma_N::AbstractMatrix{Float64}, Sigma_O::AbstractMatrix{Float64})
    
#     actions = continuousActions(n_dims)
#     return POMDP_KalmanFilter{continuousActions}(T!, T, actions,R, H, Sigma_N, Sigma_O, d_proc, d_obs, T_sim!, G, G_sim, delta)  
# end 

# function init(T!::Function, T::Function,R::Function, H::Function,
#               Sigma_N::AbstractMatrix{Float64}, Sigma_O::AbstractMatrix{Float64}, 
#               delta::Float64, n_dims::Int64, upper::AbstractVector{Float64}, lower::AbstractVector{Float64})
    
#     T_sim!, d_proc, d_obs, G_sim, G = generateFunctions(T!::Function, T::Function, H::Function,
#               Sigma_N::AbstractMatrix{Float64}, Sigma_O::AbstractMatrix{Float64})
    
#     actions = boundedActions(n_dims,upper,lower)
#     return POMDP_KalmanFilter{boundedActions}(T!, T, actions,R, H, Sigma_N, Sigma_O, d_proc, d_obs, T_sim!, G, G_sim, delta)  
# end 

function init(T!::Function, T::Function,R::Function, R_obs::Function, H::Function,
              Sigma_N::AbstractMatrix{Float64}, Sigma_O::Function, 
              delta::Float64,actions, 
              observations)#::AbstractVector{AbstractVector{Float64}}
    
    T_sim!, d_proc, d_obs, G_sim, G = generateFunctions(T!, T, H,
              Sigma_N, Sigma_O, actions, observations)
    
    A = reshape(collect(IterTools.product(actions, observations)), length(actions) * length(observations) )
    
    return POMDP_KalmanFilter(T!, T, actions, observations, A, R, R_obs, H, Sigma_N, Sigma_O, d_proc, d_obs, T_sim!, G, G_sim, delta)  
end 



end # module 