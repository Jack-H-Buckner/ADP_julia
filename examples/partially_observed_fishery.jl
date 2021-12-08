"""
This file defines function to simulate the state and beleif state of a fishery and a fishery manager
where only partial observaiton are made of the populations state. 

see the test file for a full description of the model. 
"""
module partially_observed_fishery

include("utils.jl")
include("simple_fishery.jl")
using Distributions
using StatsBase
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
function Tx(x,f,pars)
    m = pars[1]
    f = exp(rand(pars[7],1)[1])*f
    catch_ = x*(1-exp(-f/(m+f)))   
    return x*exp(-m-f) + pars[2]*x*exp(rand(pars[6],1)[1])/(1+pars[3]*x),  catch_
end
"""
sames as above but does not record catchs
"""
function Tx_nc(x,f,pars)
    m = pars[1]
    f = exp(rand(pars[7],1)[1])*f
    catch_ = x*(1-exp(-f/(m+f)))   
    return x*exp(-m-f) + pars[2]*x*exp(rand(pars[6],1)[1])/(1+pars[3]*x)
end


function profit(f, catch_ ,pars)
    return pars[4]*catch_ - pars[5]*f
end 


"""
dentify function of beleif state  given cumulants 
"""
function dentisy_B(chi,cumulants)
    @assert length(cumulants) < 5
    y = (chi-cumulants[1])/sqrt(cumulants[2])
    x = 1
    for i in 3:length(cumulants)
        x += cumulants[i]/(cumulants[2]^(i/2)*factorial(i)) * utils.hermite(y,i)

    end 
    return x*utils.phi(y)
end 



"""
importnance sample weights
"""
function weights_B(chi,cumulants)
    @assert length(cumulants) < 6
    y = (chi-cumulants[1])/sqrt(cumulants[2])
    x = 1
    for i in 3:length(cumulants)
        x += cumulants[i]/(cumulants[2]^(i/2)*factorial(i)) * utils.hermite(y,i)

    end 
    return x
end 


"""
sample from normal distribution, reweight to matchcumulants with gram-charlier
"""
function sample_B(N,cumulants)
    @assert length(cumulants) < 6
    d = Distributions.Normal(cumulants[1],sqrt(cumulants[2]))
    sample = rand(d,N)
    weights = broadcast(chi -> weights_B(chi,cumulants), sample)./N
    return sample, weights
end 


"""
proporgate error from state transition,
only acts on the samples 
"""
function T!(sample,f,pars)
    sample= broadcast(x -> log(Tx_nc(exp(x),f,pars)), sample)
    return sample
end


"""
update weights with observaitons
fishery dependent data is given a lower statistical weight than fishery independent observaitons
specified by catch_weight in [0,1]
g_yx - likelihood of log(Y_t) given log(x_t) assume it is normal 
"""
function observaitons!(weights, sample, log_Y_t, log_c_t, pars )
    y_weights = pdf.(pars[8], sample .- log_Y_t)
    c_weights = pdf.(pars[7], sample .- log_c_t)
    
    weights = weights.*y_weights.*c_weights.^pars[9]
    weights = weights ./ sum(weights)
    return weights
end 



"""
initializes a simulation of the beleif and state dynamics from 
a set of cumulants 

inputs
B - beleif state cumulants
f - fishing mortality rate
pars - paramters for state transitions

returns:
samples - particel fislter particels 
weights - particle filter weights 
B - the first length(B) cumulants 
log_x_t - true state of the system (abundacne)
pi - profit from fishery 
"""
function T_cumulants!(B,f,pars, N)
    
    samples, weights = sample_B(N,B)
    log_x_t = samples[sample(collect(1:N),StatsBase.pweights(weights),1)]
    log_x_t, log_c_t = log.(Tx(exp(log_x_t[1]),f,pars))
    pi = profit(f, exp(log_c_t) ,pars)
    samples = T!(samples,f,pars)
    log_Y_t = log_x_t + rand(pars[8],1)[1]
    weights = observaitons!(weights, samples, log_Y_t, log_c_t, pars )

    B = broadcast(degree -> utils.cumulant(samples, weights, degree), 1:length(B))
    return samples, weights, B, log_x_t, pi
end


"""
updates a set of samples represetning a beleif state 
and updates the true underlying state 

inputs
samples
weights
f - fishing mortality rate
log_x_t
pars - paramters for state transitions

returns:
samples - particel fislter particels 
weights - particle filter weights 
B - the first length(B) cumulants 
log_x_t - true state of the system (abundacne)
pi - profit from fishery 
"""
function T_samples(samples, weights, N_moments, f, log_x_t, pars)
    log_x_t, log_c_t = log.(Tx(exp(log_x_t[1]),f,pars))
    pi = profit(f, exp(log_c_t) ,pars)
    samples = T!(samples,f,pars)
    log_Y_t = log_x_t + rand(pars[8],1)[1]
    weights = observaitons!(weights, samples, log_Y_t, log_c_t, pars )

    B = broadcast(degree -> utils.cumulant(samples, weights, degree), 1:N_moments)
    return samples, weights, B, log_x_t, pi
end

"""
during longer simualtions the weights given to some particles will be come 
quite small. This reduces the efficenfy of the particle filter algorithm.  

To fix this this function draws a weighted sample with replacement, this 
process produces a new sample with all weights equal to 1/n_samples that still 
represetns the same probability distribution. 
"""
function reweight_samples!(samples, weights)
    N = length(samples)

    inds = sample(collect(1:N),StatsBase.pweights(weights),N)

    samples = samples[inds]
    weights = repeat([1/N], N)
    return samples, weights
end

using Plots
function test_sim(B,f,pars,N_steps, N)
    B_ = zeros(N_steps, length(B))
    log_x_t_ = zeros(N_steps)
    pi_ = zeros(N_steps)
    samples, weights, B, log_x_t, pi = T_cumulants!(B,f,pars,N)
    samples, weights = reweight_samples!(samples, weights)
    
    B_[1,:] = B
    log_x_t_[1] = log_x_t
    pi_[1] = pi
    
    for i in 2:N_steps
        samples, weights, B, log_x_t, pi = T_samples(samples, weights, length(B),f, log_x_t, pars)
        samples, weights = reweight_samples!(samples, weights)
        
        B_[i,:] = B
        log_x_t_[i] = log_x_t
        pi_[i] = pi
    end
    
    return B_, log_x_t_, pi_
end 

m = 0.3#pars[1] - Float64 m natrual mortality rate
a = 3.0#pars[2] - Float64 a recruitment rate
b = 0.1#pars[3] - Float64 b strength of density dependence
p = 10 #pars[4] - Float64 price of fish
c = 2.0 #pars[5] - Float64 costs of fishing
sigma = 2.0
sigma_catch = 0.1
simga_obs = 5.0
d_recruits = Distributions.Normal(0,sigma)#pars[6] - Distribution d_recruits probability of recruitment
d_catch = Distributions.Normal(0,sigma_catch) #pars[7] - Distribution d_catch probability of catch
g_yx  = Distributions.Normal(0,0.5) #pars[8] - Distribution g_yx probability of observation log_y_t given log_x_t
weight_catch = 0.05 #pars[9] - Float64 weight_catch statistical weight of fisheries dependent data 
pars = [m,a,b,p,c,d_recruits,d_catch,g_yx,weight_catch]
end 