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
            R_obs::Function,
            H::Function,
            Sigma_N::AbstractMatrix{Float64},
            Sigma_O::Function,
            delta::Float64,
            actions,
            observations,
            lower_mu,
            upper_mu)
    
    POMDP = POMDPs.init(T!, T, R, R_obs, H,Sigma_N, Sigma_O, delta, actions, observations)
    # set intermidiate
    dims_x = size(Sigma_N)[1]
    dims_y = size(Sigma_O(actions[1], observations[1]))[1]
    m_Quad_x = 5
    m_Quad_y = 5
    m_Quad = 5
    bellmanIntermidiate = BellmanOpperators.init_bellmanIntermidiate(dims_x,dims_y,m_Quad_x,m_Quad_y)
    obsBellmanIntermidiate = BellmanOpperators.init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP)
    # default to 30 grid point for observed component and 7 for uncertinaty adjustment
    grids_obs = 15
    grids_unc = 5
    valueFunction = ValueFunctions.init_adjGausianBeleifsInterp(grids_obs , grids_unc, lower_mu, upper_mu)
    kalmanFilterSolver(POMDP,bellmanIntermidiate,obsBellmanIntermidiate,valueFunction, "NA", "Two stage VFI", "Initialized")
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
    print("here")
    while (test > tol) && (iter < max_iter)
        print(iter)
        print(" ")
        print(test)
        print(" ")
        iter += 1
        i = 0
        # get updated values 
        vals0 .= vals
        for x in kalmanFilterSolver.valueFunction.baseValue.grid
            i+=1
            vals[i] = BellmanOpperators.obs_Bellman([x[1],x[2]],kalmanFilterSolver.obsBellmanIntermidiate,  
                            kalmanFilterSolver.valueFunction.baseValue, 
                            kalmanFilterSolver.POMDP)
        end 
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update_base!(kalmanFilterSolver.valueFunction,vals)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Observed model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: Observed model solved"
    else 
        kalmanFilterSolver.warnngs = "Observed model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: Observed model failed"
    end 
    
end



##############################################
###    Bellman opperator for full model    ###
##############################################

function solve(kalmanFilterSolver)
    tol = 10^-3
    max_iter = 5*10^2
    test = tol+1.0
    
    nodes = kalmanFilterSolver.valueFunction.uncertantyAdjustment.nodes
    vals = zeros(length(nodes)) 
    vals0 = zeros(length(nodes))
    tol *=  length(nodes)
    test *= length(nodes)
    
    iter = 0

    while (test > tol) && (iter < max_iter)
        print(iter)
        print(" ")
        println(test)

        iter += 1
        i = 0
        # get updated values 
        vals0 .= vals
        for s in nodes
            i+=1
            if mod(i, 100) == 0
                print(i/length(nodes))
                print(" ")
            end 
            vals[i] = BellmanOpperators.Bellman!(s, kalmanFilterSolver.bellmanIntermidiate,
                                        kalmanFilterSolver.POMDP, kalmanFilterSolver.valueFunction)
        end 
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update_adjustment!(kalmanFilterSolver.valueFunction,vals)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Full model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: full model solved"
    else 
        kalmanFilterSolver.warnngs = "Full model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: full model failed"
    end 
end 
# function solve_parallel(kalmanFilterSolver)
#     tol = 10^-4
#     max_iter = 5*10^2
#     test = tol+1.0
    
#     nodes = kalmanFilterSolver.valueFunction.uncertantyAdjustment.nodes
#     vals = zeros(length(nodes)) 
#     vals0 = zeros(length(nodes))
#     vals_shared = SharedArray{Float64}(length(nodes))
#     tol *=  length(nodes)
#     test *= length(nodes)
    
#     iter = 0

#     while (test > tol) && (iter < max_iter)
#         print(iter)
#         print(" ")
#         print(test)
#         print(" ")
#         iter += 1
#         i = 0
#         # get updated values 
#         vals0 .= vals
#         @parallel for s in nodes
#             i+=1
#             vals_shared[i] = BellmanOpperators.Bellman!(s, kalmanFilterSolver.bellmanIntermidiate,
#                                         kalmanFilterSolver.POMDP, kalmanFilterSolver.valueFunction)
#         end 
#         vals .= vals_shared
#         # update value function 
#         test = sum((vals .- vals0).^2)
#         ValueFunctions.update_adjustment!(kalmanFilterSolver.valueFunction,vals)
#     end 
    
#     if iter < max_iter
#         kalmanFilterSolver.warnngs = "Full model converged"
#         kalmanFilterSolver.algorithm = "two stage VFI: full model solved"
#     else 
#         kalmanFilterSolver.warnngs = "Full model failed"
#         kalmanFilterSolver.algorithm = "Two stage VFI: full model failed"
#     end 
# end 
end # module 