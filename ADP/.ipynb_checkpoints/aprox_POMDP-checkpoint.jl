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
        at = POMDP.Q.grid[a]
        print(" ")
        print(at)
        for j in 1:N_steps
            xt, Rt = POMDP.T!(xt,at,POMDP.POMP)
            Q_ += d*Rt
            d *= POMDP.delta
            Bt = POMDP.stat(POMDP.POMP)
            at = POMDP.Q.argmax_(POMDP.Q,Bt)
        end 

        Q_ += POMDP.Q.max_(POMDP.Q,[Bt])[1]
    end 
    return Q_/N_chains
end 

end # module