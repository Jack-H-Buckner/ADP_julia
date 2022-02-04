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

"""
mutable struct kalmanFilterSolver{T1,T2}
    POMDP::POMDPs.POMDP_KalmanFilter{T1}
    bellmanIntermidiate::BellmanOpperators.bellmanIntermidiate
    valueFunction::{T2}
end 


end # module 