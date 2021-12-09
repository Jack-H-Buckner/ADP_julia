"""
This module defines four primary methods:
filter
importance_filter
reweight
sample_grahm_charlier

filter! -  a standard particle filter algorithm the user defines a set of particls
a transition function that simualtes a time step of the stochastic process model,
an observaiton model g(y|x) and an observaiton y_t. 

importance_filter - This algorithm tries to gain a computational advantage by taking advantage 
of the fact that we know the true value of the state varible when simualting a POMP. I have not 
determined how to best do this yet, but I think I might be able to bias the draws from the transition
function around the true state and reweight them with importnace sampling.

reweight - resamples the prticles using weighted sampling w/ replacement to produce a new set of samples
with equalweights that represent the beleif state. 

sample_grahm_charlier - samples from a distribution aproximated with the gram charlier method inputs up to the
first five cummulants and a number of samples. Returns a set of samples and weights. only defined in one dimension 
for now. 


Beleif state:

B_samples- represents the beleif state this is a tuple
(samples, weights)
samples, is a vector that stors the value of the state variable
weights is log scale weights for each sample 
sum(exp.(weights)) == 1 for resampling 
"""
module particle_filter

using StatsBase
using Distributions 
include("../utils.jl")
mutable struct POMP
    samples::AbstractVector
    weights::AbstractVector{Float64}
    N::Int
    G # likelihood function
    T! # state transition simulations
end 


"""
during longer simualtions the weights given to some particles will be come 
quite small. This reduces the efficenfy of the particle filter algorithm.  

To fix this this function draws a weighted sample with replacement, this 
process produces a new sample with all weights equal to 1/n_samples that still 
represetns the same probability distribution. 
"""
function reweight_samples!(particle_POMP)
    samples, weights = particle_POMP.samples,particle_POMP.weights
    N = length(samples)
    inds = sample(collect(1:N),StatsBase.pweights(exp.(weights)),N)
    samples = samples[inds]
    weights = log.(repeat([1/N], N))
    particle_POMP.samples,particle_POMP.weights = samples, weights
    return particle_POMP
end



"""
B_samples - samples and weights
T! - transition function 
G - observaiton function G(x,y,a) = P(X=x|y,a) 
y_t - observaiton
a - action taken by decision maker
"""
function filter!(particle_POMP,y_t, a)
    samples, weights = particle_POMP.samples, particle_POMP.weights
    G, T! = particle_POMP.G, particle_POMP.T!
    samples = T!.(samples,a)
    w = log.(broadcast(x -> G(x,y_t,a), samples))
    weights += w
    particle_POMP.samples, particle_POMP.weights = samples, weights
    return particle_POMP
end



"""
GC_samples
draws N samples from a normal distribtion following the first two moments and then rewights the
sampleto match higher cumulants using the Gram-Charlier aproximation 

Thsi is for 1 dimensional dsns only 
"""
function GC_samples(N, cumulants)
    @assert length(cumulants) < 6
    d = Distributions.Normal(cumulants[1],sqrt(cumulants[2]))
    samples = rand(d,N)
    y = (samples .-cumulants[1])./sqrt(cumulants[2])
    weights = repeat([1],N)
    for i in 3:length(cumulants)
        weights += cumulants[i]/(cumulants[2]^(i/2)*factorial(i)) * utils.hermite.(y,i)
    end 
    weights[weights .<= 0] .= 10^-12
    return samples, log.(weights./N)
end


"""
reset the samples of a POMP model to 
"""
function init_samples!(POMP,N,cumulants)
    samples,weights = GC_samples(N, cumulants)
    POMP.samples,POMP.weights=samples,weights
    return POMP
end 

end # module 