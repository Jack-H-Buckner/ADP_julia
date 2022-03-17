# load libraries
using JLD2
using Plots
# load modules
include("/home/jhbuckne/ADP_julia/Examples.jl")
include("/home/jhbuckne/ADP_julia/ValueFunctions.jl")
include("/home/jhbuckne/ADP_julia/BeleifMDPSolvers.jl")
include("/home/jhbuckne/ADP_julia/BellmanOpperators.jl")
# load data
@load "VF_dat.jld2" VF_dat


pars = Examples.unknown_growth_rate_pars

# load function and parameters from Examples
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



# set solver 
solver1 = BeleifMDPSolvers.init(T_!, T_, R, R_obs, Examples.H, Sigma_N, Sigma_O,delta,actions,observations,lower,upper;
            m_Quad_x = 7, m_Quad_y = 7,m_Quad = 8,
            n_grids_obs = 20,n_grid = 5)

# update value function 
ValueFunctions.update_interpolation!(solver1.valueFunction.baseValue, VF_dat[1])
ValueFunctions.update_interpolation!(solver1.valueFunction.uncertantyAdjustment.chebyshevInterpolation, VF_dat[2])


# plot value function 
n_grid = 20
sol = solver1
act = broadcast(x -> BellmanOpperators.obs_Policy(x, sol.obsBellmanIntermidiate[1],sol.valueFunction.baseValue, sol.POMDP, sol.optimizer),
solver1.valueFunction.baseValue.grid)
act = broadcast(x -> x[1],act)
act = reshape(act, n_grid,n_grid)
x = reshape(broadcast( x ->x[1], sol.valueFunction.baseValue.grid),n_grid,n_grid)
p = Plots.plot(exp.(x[:,1]), act[:,1], label = "1", legend =:topleft,color = RGBA(1/n_grid,1-1/n_grid,1,0.5))
for i in 2:n_grid
    Plots.plot!(p,exp.(x[:,n_grid]),act[:,i], label = string(i),color = RGBA(i/n_grid,1-i/n_grid,1,0.5))
end 

savefig(p,"base_value_function.png")
