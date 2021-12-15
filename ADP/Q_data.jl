"""
In aproximate dynamic programming the value function is represetned by 
data generated in each simulation as it is by the function aproximation tool.
In this module I define a mutable struct and some methods which store the data
produced in the simualtions and return data from the current and old simuation 
that can be used to fit the function aproximaiton. 
"""
module Q_data
using StatsBase
mutable struct data
    N::Int64 # expected length of buffer + 2%
    M::Int64 # number of samples 
    s::AbstractArray{} # states
    a::AbstractArray{} # actions
    Q_::AbstractArray{} # estimated value
end 

"""
N_sim - number of simulations per update
delta - fraction of data points saved each iteration 
s_dims - dimensions of the state space
a_dims - dimensions of that aciton space (1 for discrete)
"""
function init_Q_data(N_sim, delta, s_dims, a_dims)
    N = N_sim/(1-delta) 
    N *= 1.02
    N = floor(Int,N)
    s = zeros(N,s_dims)
    a = zeros(N,a_dims)
    Q_ = zeros(N)
    return data(N,1,s,a,Q_)
end 


function sample_data!(data, delta)
    M = floor(Int,data.M*delta)
    print(M)
    print(" ")
    if M == 0
        M = 1
    end
    inds = StatsBase.sample(collect(1:data.M),M)
    data.s[1:M,:] = data.s[inds,:]
    data.s[M:end,:] .= 0
    data.a[1:M,:] = data.a[inds,:]
    data.a[M:end,:] .= 0
    data.Q_[1:M,:] = data.Q_[inds]
    data.Q_[M:end,:] .= 0
    data.M = M
end 


function add_data!(data, Q_, s, a)
    M_new = length(Q_)
    
    print(data.M)
    print(" ")
    if (M_new + data.M) < data.N
        data.s[(data.M+1):(data.M+M_new),:] = s
        data.a[(data.M+1):(data.M+M_new),:] = a
        data.Q_[(data.M+1):(data.M+M_new)] = Q_
    else
        extra = (M_new + data.M) - data.N + 1
        data.N += extra
        print(extra)
        data.s = vcat(data.s ,zeros(extra, size(data.s)[2])) 
        data.a = vcat(data.a ,zeros(extra, size(data.a)[2])) 
        data.Q_ = vcat(data.Q_, zeros(extra)) 
        data.s[(data.M+1):(data.M+M_new),:] = s
        data.a[(data.M+1):(data.M+M_new),:] = a
        data.Q_[(data.M+1):(data.M+M_new)] = Q_
    end 
    data.M = data.M+ M_new
end


end # module