"""
Uses montecarlo methods to evaluate bellman operator for continuous state POMDPs


"""
module MC_POMDP

include("../POMP/particle_filter.jl")
include("Q_data.jl")
include("chebyshev_Q.jl")

using Distributions
using StatsBase
mutable struct POMDP
    POMP # POMP object from particle_filter.jl 
    stat # summary statistic of the POMP samples nd weigths
    init_samples! # draw samples for POMP based on beleif state 
    data # object to store samples from simulations from Q_data.jl
    Q # Q funciton aproximation from chebyshev_Q.jl
    delta # discount factor
end 



"""
evaluates the bellman opperator using MC integration to evlauate the expecataitons and 
a particle filter simulate the state transition. 
"""
function MC_Bellman(B,a,POMDP,N_particles, N_state_transitions)
    
    # initlaize particles from beleif state data 
    POMDP.init_samples!(POMDP.POMP, N_particles,B)
    samples, weights = POMDP.POMP.samples, exp.(POMDP.POMP.weights)
    print(a)
    # simulate rewards and state transitions 
    at = POMDP.Q.grid[a]
    out = broadcast(x -> POMDP.POMP.T!(x,at), POMDP.POMP.samples) # simulate step for each particle 
    s = broadcast(i->out[i][1], 1:N_particles) # get updated states 

    R = sum(broadcast(i->out[i][2].*weights[i], 1:N_particles)) # calculate expected rewards 
    
    # sample simulated observaitons 
    xt = particle_filter.sample_x(POMDP.POMP,N_state_transitions)
    
    yt = broadcast(x -> POMDP.POMP.G_sim(x,at),xt) # make observations

    V = R 

    #print(broadcast(y -> length(y), yt))
    for i in 1:N_state_transitions
        weights
      
        # get weights  for y_t[i]

        w = broadcast(x -> POMDP.POMP.G(x,yt[i],at), samples)
        if isnan(POMDP.Q.max_(POMDP.Q,POMDP.stat(samples, weights .* w))[1])
            print(yt)
        end 
        V += POMDP.Q.max_(POMDP.Q,POMDP.stat(samples, weights .* w))[1]/N_state_transitions
        
    end 
    
    return V
end 



"""
evaluates the bellman opperator using MC integration to evlauate the expecataitons and 
a particle filter simulate the state transition. 
"""
function opt_MC_Bellman(B,p,POMDP, N_particles, N_state_transitions)
    
    # select action based on Q function
    a = 0
    if rand(1)[1] < p
        a = StatsBase.sample(POMDP.Q.A, 1)
    else
        a = POMDP.Q.argmax_ind(POMDP.Q,B)
    end 
    a = a[1]
    at = POMDP.Q.grid[a]
    # initlaize particles from beleif state data 
    POMDP.init_samples!(POMDP.POMP, N_particles,B)
    samples, weights = POMDP.POMP.samples, exp.(POMDP.POMP.weights)

    # simulate rewards and state transitions 
    out = broadcast(x -> POMDP.POMP.T!(x,at), POMDP.POMP.samples) # simulate step for each particle 
    s = broadcast(i->out[i][1], 1:N_particles) # get updated states 

    R = sum(broadcast(i->out[i][2].*weights[i], 1:N_particles)) # calculate expected rewards 
    
    # sample simulated observaitons 
    xt = particle_filter.sample_x(POMDP.POMP,N_state_transitions)
    
    yt = broadcast(x -> POMDP.POMP.G_sim(x,at),xt) # make observations

    V = R 

    #print(broadcast(y -> length(y), yt))
    for i in 1:N_state_transitions
        weights
      
        # get weights  for y_t[i]

        w = broadcast(x -> POMDP.POMP.G(x,yt[i],at), samples)
        if isnan(POMDP.Q.max_(POMDP.Q,POMDP.stat(samples, weights .* w))[1])
            print(yt)
        end 
        V += POMDP.Q.max_(POMDP.Q,POMDP.stat(samples, weights .* w))[1]/N_state_transitions
        
    end 
    
    return V, a
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
function step!(POMDP,N_samples, alpha, N_particles, N_transitions)
    s, a = uniform_sample_sa(POMDP, N_samples)
    Q_ = broadcast(i-> MC_Bellman(s[i],a[i],POMDP, N_particles, N_transitions),1:N_samples)
 
    Q_data.sample_data!(POMDP.data, alpha)

    a = broadcast(i -> [a[i]], 1:length(a))
    Q_data.add_data!(POMDP.data, Q_, s, a)

    s,a,Q_ = Q_data.return_data!(POMDP.data)
    chebyshev_Q.update_Q_discrete!(POMDP.Q,a,s,Q_)
end 


"""
N_samples - numner of data points to add
alpha - proportion of data points to sample. 
"""
function opt_step!(POMDP,N_samples, p, alpha, N_particles, N_transitions)
    s, a = uniform_sample_sa(POMDP, N_samples)
    out = broadcast(i-> opt_MC_Bellman(s[i],p,POMDP, N_particles, N_transitions),1:N_samples)
 
    Q_ = broadcast(i-> out[i][1],1:N_samples)
    a = broadcast(i-> [out[i][2]],1:N_samples)
    
    Q_data.sample_data!(POMDP.data, alpha)
    
    Q_data.add_data!(POMDP.data, Q_, s, a)

    s,a,Q_ = Q_data.return_data!(POMDP.data)
    chebyshev_Q.update_Q_discrete!(POMDP.Q,a,s,Q_)
end 


end # module 