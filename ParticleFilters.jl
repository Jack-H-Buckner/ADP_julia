"""
This module defines a minimal set of methods for implementing particle filters
focused on solving POMDP problems. 

The structs are 
ParticleFilter - stores data for a continuous state particle filter
HiddenParticleFilter - store data for a particle filter with discrete hidden states

The functions are 
timeUpdate - Updates the particle to account for the underlying dynamics 
timeUpdate! - Inplace version of above
bayesUpdate - Updates the particles to account for a new observation
bayesUpdate! - inplace version of above

mean - the weigthed mean of the particles
cov - the covarinace mtri of the particles
cumulant - for 1d distribtuions ths calcualtes the higher moments
"""
module ParticleFilters

using StatsBase


"""
Stores samples and weights for particle filter. 
"""
mutable struct ParticleFilter
    samples::AbstractVector{AbstractVector{Float64}}
    weights::AbstractVector{Float64}
    N::Int64
    d::Int64
end 


"""
Stores samples and weights for mixed obserability particle filter

When some state are observed exactly we need 
"""
mutable struct ParticleFilterMOMDP
    x0::AbstractVector{Float64} # observed past time step
    xt::AbstractVector{Float64} # observed state current time step
    samplesH0::AbstractVector{AbstractVector{Float64}} # hidden states past time step
    samplesHt::AbstractVector{AbstractVector{Float64}} # hidden states current time step
    weights::AbstractVector{Float64} # weights
    N::Int64 # number of samples 
    dH::Int64 # 
    dx::Int64
end 

"""
initalizes a particle filter object
N - number of samples
d - number of dimensions for samples 
"""
function init(N::Int64,d::Int64)
    samples = broadcast(i->zeros(d),1:N)
    weights = repeat([1.0/N],N)
    return ParticleFilter(samples,weights,N,d)
end 

# function init(N::Int64,d::Int64,d2::Int64)
#     print("Initalizing a standard particle filter")
#     samples = broadcast(i->zeros(d,d2),1:N)
#     weights = repeat([1.0/N],N)
#     return ParticleFilter(samples,weights,N,d)
# end 

function init(N::Int64,dH::Int64,dx::Int64)

    samples = broadcast(i->zeros(d),1:N)
    weights = repeat([1.0/N],N)
    return ParticleFilter(samples,weights,N,d)
end 

"""
During longer simualtions the weights given to some particles will be come 
quite small. This reduces the efficeny of the particle filter algorithm.  

To fix this this function draws a weighted sample with replacement, this 
process produces a new sample with all weights equal to 1/n_samples that still 
represetns the same probability distribution. 
"""
function resample!(ParticleFilter)
    inds = sample(collect(1:ParticleFilter.N),StatsBase.pweights(ParticleFilter.weights),ParticleFilter.N)
    for i in 1:ParticleFilter.N
        ParticleFilter.samples[i] .= ParticleFilter.samples[inds[i]]
    end 
    ParticleFilter.weights = repeat([1/ParticleFilter.N], ParticleFilter.N)
end


"""
Updates the particles in the filter to account for the state transition. 

This is the definition of the method when only the state transition functon is provided. 
An alternative method is given below for models with mixed observability. 

ParticleFilter - a ParticleFilter object
T - state transition function  T(x,a)
a - action taken by decision maker or other auxiliary paramters 
"""
function time_update!(ParticleFilter,T!,a)
    broadcast(x -> T!(x,a),ParticleFilter.samples)
end


"""
Updates the particles in the filter to account for the state transition. 
ParticleFilter - a ParticleFilter object
G - Likelihood funciton G(y,x)
yt - observation
a - action or auxiliary paramters
"""
function bayes_update!(ParticleFilter,G,yt,a)
    ParticleFilter.weights .*= broadcast(x -> G(yt,x,a), ParticleFilter.samples)
    ParticleFilter.weights .*= 1/sum(ParticleFilter.weights) 
end

function bayes_update(ParticleFilter,G,yt,a)
    weights = ones(ParticleFilter.N)./ParticleFilter.N
    #println(broadcast(x -> G(yt,x,a)[1], ParticleFilter.samples))
    weights .*= broadcast(x -> G(yt,x,a)[1], ParticleFilter.samples)
    weights .*= 1/sum(weights) 
    return ParticleFilter.samples, weights 
end


######################################################
### Methods for problems with mixed observability  ###
######################################################

"""
Time update  MOMP
ParticleFilter - The particle filter object to update
T - hidden state transition function
f - likelihood of 
"""
function time_update!(ParticleFilter,T,f, x0::AbstractVector{Float64}, xt::AbstractVector{Float64},a)
    broadcast(H -> T!(H,x0,a),ParticleFilter.samples) # simulate time step of unobserved states 
    ParticleFilter.weights .*= broadcast(H -> f(xt,H,x0,a), ParticleFilter.samples) # weight samples to account for observed state
    ParticleFilter.weights .*= 1/sum(ParticleFilter.weights) # normalize weigths
end





end # module