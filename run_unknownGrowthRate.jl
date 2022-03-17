# test loading libraries
import Pkg
Pkg.add("Distributions")
Pkg.add("Roots")
Pkg.add("KalmanFilters")
Pkg.add("LinearAlgebra")
Pkg.add("IterTools")
Pkg.add("FastGaussQuadrature")
Pkg.add("StatsBase")
Pkg.add("JLD2")

#import libraries
using JLD2

# load modules 
include("Examples.jl")
include("ValueFunctions.jl")
include("BeleifMDPSolvers.jl")


# set parameters 
pars = Examples.unknown_growth_rate_pars

T_! = (x,f) -> Examples.unknown_growth_rate_T!(x,f,pars)
T_ = (x,f) -> Examples.unknown_growth_rate_T(x,f,pars)
R = (x,f,obs) -> Examples.unknown_growth_rate_R(x,f,obs,pars)
R_obs = (x,f) -> Examples.unknown_growth_rate_R_obs(x,f,pars)
Sigma_N = Examples.Sigma_N
H = (x,a,obs) -> Examples.H * x
Sigma_O = (a,obs) -> reshape(Examples.Sigma_O(obs),1,1)
delta = 0.95
actions = Examples.unknown_growth_actions
observations = Examples.unknown_growth_observations

upper = Examples.unknown_growth_upper
lower = Examples.unknown_growth_lower

# define solver 
solver1 = BeleifMDPSolvers.init(T_!, T_, R, R_obs, Examples.H, Sigma_N, Sigma_O,delta,actions,observations,lower,upper;
            m_Quad_x = 7, m_Quad_y = 7,m_Quad = 9,
            n_grids_obs = 20,n_grid = 6)

# solve observed system 
BeleifMDPSolvers.solve_observed_parallel(solver1)

# save output
save_object("data/unknownGrowthGateSolver.jld2", solver1)
