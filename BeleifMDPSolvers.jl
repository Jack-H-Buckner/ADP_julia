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
mutable struct kalmanFilterSolver
    POMDP
    bellmanIntermidiate::BellmanOpperators.bellmanIntermidiate
    obsBellmanIntermidiate::BellmanOpperators.obsBellmanIntermidiate
    valueFunction::ValueFunctions.adjGausianBeleifsInterp
    policyFunction
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
            actions,
            observations,
            lower_mu,
            upper_mu)
    
    POMDP = POMDPs.init(T!, T, R, H,Sigma_N, Sigma_O, delta, actions, observations)
    # set intermidiate
    dims_x = size(Sigma_N)[1]
    dims_y = size(Sigma_O)[1]
    m_Quad_x = 5 
    m_Quad_y = 5
    m_Quad = 10
    bellmanIntermidiate = BellmanOpperators.init_bellmanIntermidiate(a0,dims_x,dims_y,m_Quad_x,m_Quad_y)
    obsBellmanIntermidiate = BellmanOpperators.init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP)
    # default to 30 grid point for observed component and 7 for uncertinaty adjustment
    valueFunction = ValueFunctions.init_adjGausianBeleifsInterp(30, 5, lower_mu, upper_mu)
    kalmanFilterSolver(POMDP,bellmanIntermidiate,obsBellmanIntermidiate, "Two stage VFI", "Initialized")
end 




##############################################
### Bellman opperator for observed systems ###
##############################################


"""
    solve_observed(kalmanFilterSolver)

Solves the dynamic program for the fully observed version of the model using
value function iteratation over a set of nodes used in the funciton aproximation. 

Currently this only supports methods that use the adjGausianBeleifsInterp value 
function from the ValueFunctions.jl module. 
"""
function solve_observed(kalmanFilterSolver)
    tol = 10^-5
    max_iter = 5*10^2 
    test = tol+1.0
    
    nodes = kalmanFilterSolver.valueFunction.baseValue.grid 
    vals = zeros(length(nodes))
        
    vals0 = zeros(length(nodes))
        
    tol *=  length(nodes)
    test *= length(nodes)
    
    iter = 0
    while (test < tol) && (iter < max_iter)
        iter += 1
        i = 0
        # get updated values 
        vals0 .= vals
        for x in kalmanFilterSolver.valueFunction.baseValue.grid
            i+=1
            vals[i] = obs_Bellman!(kalmanFilterSolver.obsBellmanIntermidiate, [x[1],x[2]],kalmanFilterSolver.valueFunction.baseValue, kalmanFilterSolver.POMDP)
        end 
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update_base!(kalmanFilterSolver.valueFunction,v_mu)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Observed model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: Observed model solved"
    else 
        kalmanFilterSolver.warnngs = "Observed model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: Observed model failed"
    end 
    
end


end # module 