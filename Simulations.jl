module Simulations

using Distributions
using KalmanFilters
using LinearAlgebra

include("BellmanOpperators.jl")
function simulate_obs(x0, solver, T)
    x = broadcast(i -> zeros(length(x0)), 1:T)
    pi = zeros(T)
    action = broadcast(i -> zeros(solver.policyFunction.observationDims), 1:T)
    xt = x0
    for i in 1:T
        x[i] .= xt
        #act = broadcast( i -> solver.policyFunction.actionPolynomials[i].baseValue(xt), 1:solver.policyFunction.actionDims)
        act = BellmanOpperators.obs_Policy(xt, 
                                        solver.obsBellmanIntermidiate[Threads.threadid()],
                                        solver.valueFunction.baseValue, 
                                        solver.POMDP, solver.optimizer)
        
        pit = solver.POMDP.R_obs(xt,act)
        xt = solver.POMDP.T_sim!(xt,act)

        pi[i] = pit
        action[i] = act
    end 
    
    return x, pi, action
    
end

function simulate_obs_pf(x0, solver, T)
    x = broadcast(i -> zeros(length(x0)), 1:T)
    pi = zeros(T)
    action = broadcast(i -> zeros(solver.policyFunction.observationDims), 1:T)
    xt = x0
    for i in 1:T
        x[i] .= xt
        act = broadcast( i -> solver.policyFunction.actionPolynomials[i].baseValue(xt), 1:solver.policyFunction.actionDims)
#         act = BellmanOpperators.obs_Policy(xt, 
#                                         solver.obsBellmanIntermidiate[Threads.threadid()],
#                                         solver.valueFunction.baseValue, 
#                                         solver.POMDP, solver.optimizer)
        
        pit = solver.POMDP.R_obs(xt,act)
        xt = solver.POMDP.T_sim!(xt,act)

        pi[i] = pit
        action[i] = act
    end 
    
    return x, pi, action
    
end


function simulate_obs_random(x0, solver, T)
    x = broadcast(i -> zeros(length(x0)), 1:T)
    pi = zeros(T)
    action = broadcast(i -> zeros(solver.policyFunction.observationDims), 1:T)
    xt = x0
    for i in 1:T
        x[i] .= xt
        #act = broadcast( i -> solver.policyFunction.actionPolynomials[i].baseValue(xt), 1:solver.policyFunction.actionDims)
        act = rand(solver.POMDP.actions.actions)
        
        pit = solver.POMDP.R_obs(xt,act)
        xt = solver.POMDP.T_sim!(xt,act)

        pi[i] = pit
        action[i] = act
    end 
    
    return x, pi, action
    
end



function value_obs(x0, solver, N)
    acc = 0
    delta = solver.POMDP.delta
    T = floor(Int, -2.0 * log(10)/log(delta))
    xt = broadcast(i -> copy(x0), 1:N)#repeat([x0],N)
    Threads.@threads for i in 1:N
        x, pi, action = simulate_obs(xt[i], solver, T)

        acc += sum( pi .* delta.^(1:T))
            
    end
        
    return acc / N
end

function value_obs_pf(x0, solver, N)
    acc = 0
    delta = solver.POMDP.delta
    T = floor(Int, -2.0 * log(10)/log(delta))
    xt = broadcast(i -> copy(x0), 1:N)#repeat([x0],N)
    Threads.@threads for i in 1:N
        x, pi, action = simulate_obs_pf(xt[i], solver, T)

        acc += sum( pi .* delta.^(1:T))
            
    end
        
    return acc / N
end


function value_obs_random(x0, solver, N)
    acc = 0
    delta = solver.POMDP.delta
    T = floor(Int, -2.0 * log(10)/log(delta))
    xt = broadcast(i -> copy(x0), 1:N)#repeat([x0],N)
    Threads.@threads for i in 1:N
        x, pi, action = simulate_obs_random(xt[i], solver, T)

        acc += sum( pi .* delta.^(1:T))
            
    end
        
    return acc / N
end



function simulate_2d(s0, T, solver)
    s = []
    x = broadcast(i -> [0.0,0.0], 1:T)
    pi = zeros(T)
    action = broadcast(i -> zeros(solver.policyFunction.actionDims), 1:T)
    observation = broadcast(i -> zeros(solver.policyFunction.observationDims), 1:T)
    
    xt = reshape(rand(Distributions.MvNormal(s0[1], s0[2]),1),2)
    st = s0
    
    for i in 1:T
        act, obs = solver.policyFunction(st)
        pit = solver.POMDP.R(xt,act,obs)
           
        push!(s, st)
        x[i] = xt
        action[i] = act
        observation[i] = obs
        pi[i] = pit
        
      
        
        xt = solver.POMDP.T_sim!(xt,act)
        
        x_hat, x_cov = st
        tu = KalmanFilters.time_update(x_hat, x_cov, x ->solver.POMDP.T(x,act),  solver.POMDP.Sigma_N)
        x_hat, x_cov = KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
        
        yt = solver.POMDP.H * xt 
           
      
        yt .+= reshape(rand(Distributions.MvNormal(zeros(length(yt)), solver.POMDP.Sigma_O(act, obs)),1),length(yt))
        
        mu = KalmanFilters.measurement_update(x_hat, x_cov,yt,solver.POMDP.H,solver.POMDP.Sigma_O(act, obs))
        st = KalmanFilters.get_state(mu), KalmanFilters.get_covariance(mu)
        
        
    end 
    
    
    return s,x,pi,action,observation
end




end 