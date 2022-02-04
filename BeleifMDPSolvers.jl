"""
    Algorithms

Initialize:
- value function
- POMDP object
- Bellman intermidiate

Algorithms:
gridded VFI 
Monte Carlo VFI

Policy iteration 
ADP chain (Policy iteration)
"""
module BeleifMDPSolvers


include("ValueFunctions.jl")
include("BellmanOpperators.jl")
include("POMDPs.jl")

"""
    kalmanFilterSolver

This object stores all of the data required to solve a beleif state MDP 
representing the beleif dynamics with an unscented kalman filter. This includes
a POMDP object that stores the information required to solve the problem a
bellman opperator intermidiate that is used to improve the performance of the
algorith by allowing many operatios to be done inplace, a value funtion 
object, and a policy function object. In addition to these primary objects that 
are used to solve the object also stores data on the performance of the algorith as
strings under the algorithm and warnings 

"""
mutable struct kalmanFilterSolver{T1,T2,T3}
    POMDP::POMDPs.POMDP_KalmanFilter{T1}
    bellmanIntermidiate::BellmanOpperators.bellmanIntermidiate
    valueFunction::{T2}
    policyFunction::{T3}
    algorithm::String
    warnngs::String
end 



"""
    init


The kalman filter POMDP problem is defined with several components:

A state transition function T. For the Kalman filter algorithm this is 
defined with three user inputs, to representatins of the deterministic state 
transition function and a covariance matrix.

T - the transition function defines the expected future state given a state action pair:  T(x,a) = E[x']
T! - inplace verison of the transition function:  T!(x,a) -> x = E[x']
Sigma_N - The covarinace of the process noise (which is assumed to be gausian)


The reward function R. This represents the expected rewards given a state action pair
(x,a) it is defined as a function by the user. 
R - The reward functon maps a state action pair to the within period profits 

The observation model describes the likelihood of an observaiton y given a state aciton pair
(x,a). It is defined by a deterministic observaiton function and a covariance matrix.  
H - The observaiton model:  H(x,a) = E[y]
Sigma_O - the covariance of the observaiton noise (which is assumed to be gausian)

The action space A. This is represented by discrete set of alternatives
a bounded case of  


Sigma_N - The covarinace of the process noise (which is assumed to be gausian)
"""
function init(T!::Function,
            T::Function, 
            R::Function, 
            H::Function
            Sigma_N::AbstractMatrix{Float64}
            Sigma_O::AbstractMatrix{Float64},
            delta::Float64, 
            upper_s::AbstractVector{Float64},
            lower_s::AbstractVector{Float64},
            a0)
    
    POMDP = POMDPs.init(T!, T, R, H,Sigma_N, Sigma_O, delta, n_dims)
    # set intermidiate
    a0 = (upper_a .+ lower)./2 
    dims_x = size(Sigma_N)[1]
    dims_y = size(Sigma_O)[1]
    m_Quad_x = 5 
    m_Quad_y = 5
    
    bellmanIntermidiate = BellmanOpperators.init_bellmanIntermidiate(a0,dims_x,dims_y,m_Quad_x,m_Quad_y)
    
    # set value function 
    m = 10
    valueFunction = ValueFunctions.init_interpolation(lower_s,upper_s,m)
end 




function VFI(kalmanFilterSolver)
    
end 


end # module 