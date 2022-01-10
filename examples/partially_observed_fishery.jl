"""
This file defines function to simulate the state and beleif state of a fishery and a fishery manager
where only partial observaiton are made of the populations state. 

see the test file for a full description of the model. 
"""
module partially_observed_fishery

include("../utils.jl")
include("../POMP/particle_filter.jl")
include("simple_fishery.jl")
using Distributions
using StatsBase


"""
single period rewards
"""
function profit(f, catch_ ,pars)
    return pars[4]*catch_ - pars[5]*f
end 



"""
populaitn dynamics 

pars[1] - Float64 m natrual mortality rate
pars[2] - Float64 a recruitment rate
pars[3] - Float64 b strength of density dependence
pars[4] - Float64 price of fish
pars[5] - Float64 costs of fishing
pars[6] - Distribution d_recruits probability of recruitment
pars[7] - Distribution d_catch probability of catch
pars[8] - Distribution g_yx probability of observation log_y_t given log_x_t
pars[9] - Float64 weight_catch statistical weight of fisheries dependent data 
"""
function T!(x,f,pars)
    x = x[1]
    x = exp(x)
    m = pars[1]
    f = f
    catch_ = x *(1-exp(-f/(m+f)))*exp(rand(pars[7],1)[1])  
    x = x*exp(-m-f)
    x += pars[2]*x*exp(rand(pars[6],1)[1])/(1+pars[3]*x))
    return [log(x)],  profit(f, catch_ ,pars)
end


"""
log likelihood of x given 
"""
function G(x,y,f,pars)
    x = x[1]
    m = pars[1]
    log_Y_t, log_c_t = y[1], y[2]
    L_y = pdf.(pars[8], x - log_Y_t)
    L_c = pdf.(pars[7], x - (log_c_t - log(1-exp(-f/(m+f)))))
    L = L_y*L_c^pars[9]
    return L
end


"""
log likelihood of x given 
"""
function G_sim(x,f,pars)
    m = pars[1]
    x = x[1]
    log_Y_t, log_c_t = x + rand(pars[8],1)[1],  x + rand(pars[7],1)[1]+log(1-exp(-f/(m+f)))
    return [log_Y_t, log_c_t]
end



m = 0.15#pars[1] - Float64 m natrual mortality rate
a = 1.25#pars[2] - Float64 a recruitment rate
b = 0.1#pars[3] - Float64 b strength of density dependence
p = 10 #pars[4] - Float64 price of fish
c = 2.0 #pars[5] - Float64 costs of fishing
sigma = 0.2
sigma_catch = 0.1
simga_obs = 5.0
d_recruits = Distributions.Normal(0,sigma)#pars[6] - Distribution d_recruits probability of recruitment
d_catch = Distributions.Normal(0,sigma_catch) #pars[7] - Distribution d_catch probability of catch
g_yx  = Distributions.Normal(0,0.5) #pars[8] - Distribution g_yx probability of observation log_y_t given log_x_t
weight_catch = 0.05 #pars[9] - Float64 weight_catch statistical weight of fisheries dependent data 
pars = [m,a,b,p,c,d_recruits,d_catch,g_yx,weight_catch]
end 