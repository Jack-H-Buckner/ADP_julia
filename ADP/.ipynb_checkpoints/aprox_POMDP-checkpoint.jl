"""
defines a mutable struct with information required to solve a POMDP using
apriximate dynamic progamming techniques

defines methods to evaluate the bellman operator using forward iterations

currently the code only include methods for Q learning techniues. 
"""
module aprox_POMDP
include("../POMP/particle_filter.jl")
include("Q_data.jl")
include("chebyshev_Q.jl")
using StatsBase
using Distributions
"""
an object to solve POMDP's that uses particel filter to simulate the beleif dynamics
a Q funciton with discrete action space and cheushev polynomial aproximation 

POMP - partially observable markov process model from particle_filter.jl 
stat - summary statistic of the POMP samples
sample - draw samples for POMP based on beleif state 
data - object to store samples from simulations from Q_data.jl
Q - Q funciton aproximation from chebyshev_Q.jl
delta - discount factor
"""
mutable struct POMDP
    POMP # partially observable markov process model from particle_filter.jl 
    T! # transition funciton 
    stat # summary statistic of the POMP samples
    init_samples! # draw samples for POMP based on beleif state 
    init_x # draw single obersvation from samples
    data # object to store samples from simulations from Q_data.jl
    Q # Q funciton aproximation from chebyshev_Q.jl
    delta # discount factor
end 


"""
B - initial beleif state (fist n cumulants)
a - action
POMPD - model
N_particles - number of particles used to simulate beleif dynamics
N_steps - number of steps in the chain
N_chains - number of chains used to apporximate belman operator
"""
function forward_bellman_sample(B,a,POMDP, N_particles, N_steps, N_chains)
    Q_ = 0
    for i in 1:N_chains
        Bt = B
        POMDP.init_samples!(POMDP.POMP,N_particles,Bt)
        xt = POMDP.init_x(POMDP.POMP)
        d = 1.0 
        at = a
        for j in 1:N_steps
            xt, Rt = POMDP.T!(xt,at,POMDP.POMP)
            Q_ += d*Rt
            d *= POMDP.delta
            Bt = POMDP.stat(POMDP.POMP)
            at = POMDP.Q.argmax_(POMDP.Q,Bt)
        end 

        Q_ += POMDP.Q.max_(POMDP.Q,Bt)[1]
    end 
    return Q_/N_chains
end 



function uniform_sample_sa(POMDP, N_samples)
    a = POMDP.Q.a
    b = POMDP.Q.b
    B = broadcast(x -> (rand(Distributions.Uniform(0,1),2).* (b.-a) .+ a ), 1:N_samples)
    A = StatsBase.sample(POMDP.Q.A, N_samples)
    A = broadcast(i -> POMDP.Q.grid[A[i]],  1:N_samples)
    return B, A
end 

"""
N_samples - numner of data points to add
alpha - proportion of data points to sample. 
"""
function step!(POMDP, N_samples, alpha,N_particles, N_steps, N_chains)
    s, a = uniform_sample_sa(POMDP, N_samples)
    Q_ = broadcast(i->forward_bellman_sample(s[i],a[i],POMDP, N_particles, N_steps, N_chains),1:N_samples)
    Q_data.sample_data!(POMDP.data, alpha)

    a = broadcast(i -> [a[i]], 1:length(a))
    Q_data.add_data!(POMDP.data, Q_, s, a)

    s,a,Q_ = Q_data.return_data!(POMDP.data)
    chebyshev_Q.update_Q_discrete!(POMDP.Q,a,s,Q_)
end 


end # module