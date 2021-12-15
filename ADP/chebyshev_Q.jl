"""
This module provides some methods to store aproximations to the value function 
In particular these methods are designed to compute the value of the post decision state. 
These are often called Q function in the machine learning literiture. The Idea here is the 
function returns the value of being in state s and taking aciton a. This provides and advantage
when computing the bellman operator requires computing complex expectations. using a Q function 
allows the optimizaiton routine to only evaluate the Q function, no integrals are required. 
"""
module chebyshev_Q

include("../function_aproximation/chebyshev/OLS_chebyshev.jl")
"""
Q funciton wit discrete actions. This is good for
problems with small actions states, particularly is the actions
are qualitativley differnt. 
"""
mutable struct Q_discrete
    A::AbstractVector{Int64} # action space 
    grid::AbstractArray{}
    polynomials
    evaluate
    argmax_
    max_
end 


# evaluation methods       
function evaluate_Q_discrete(Q,s,a)
    return OLS_chebyshev.evaluate_polynomial(s,Q.polynomials[a])
end 


"""
returns the action that produces the maximum reward  
"""
function argmax_A_discrete(Q,s)
    V = broadcast(x -> evaluate_Q_discrete(Q,s,x), Q.A)
    a = argmax(V)
    return Q.grid[a]
end 
"""
returns the values of taking action that produces the maximum reward  
"""
function max_A_discrete(Q,s)
    V = broadcast(x -> evaluate_Q_discrete(Q,s,x), Q.A)
    a = argmax(V)
    return V[a]
end 




"""
initializes a Q fucntion with a discrete aciton space.

a, b - the lower and upper bounds for the interpolation in each dimension
m - the order of the polynomial aproximaiton
N_actions - the number of discrete actions
grid - the actions associated with each level 1:N_actions
"""
function init_Q_discrete(a,b,m,N_actions,grid)
    polynomials = []
    for i in 1:N_actions
        push!(polynomials, OLS_chebyshev.init_polynomial(a,b,m))
    end 
    return Q_discrete(collect(1:N_actions), grid, polynomials, evaluate_Q_discrete, argmax_A_discrete, max_A_discrete)
end
    
    
    
    
    
    
    









"""
Q funciton with continuous action space. This is good for problems with
small numbers of actions (1 or maybe 2) with variable intensitities. 

actions dimenstions are assumed to be the first 1-d
"""
mutable struct Q_continuous
    action_dims::AbstractVector{Int64} # specify which dims are actions and which are states
    polynomial
end 

"""
Q funciton with continuous and discrete action space. probably best if there is one continuous 
action and two of three levels of discrete actions. 
"""
mutable struct Q_mixed
    A::AbstractVector{Int64} # discrete action space 
    grid::AbstractArray{} 
    action_dims::AbstractVector{Int64} # specify which dims are actions and which are states
    polynomials
end 







"""
initializes a Q fucntion with a discrete aciton space.

a, b - the lower and upper bounds for the interpolation in each dimension
m - the order of the polynomial aproximaiton
action_dims - the dimesnion(s) associated with actions rather than states
"""
function init_Q_continuous(a,b,m,action_dims)
    poly = OLS_chebyshev.init_polynomial(a,b,m)
    return Q_continuous(action_dims, poly)
end

"""
initializes a Q fucntion with a mixed aciton space.

a, b - the lower and upper bounds for the interpolation in each dimension
m - the order of the polynomial aproximaiton
N_actions - the number of discrete actions
grid - the actions associated with each level 1:N_actions
action_dims - the dimesnion(s) associated with actions rather than states
"""
function init_Q_mixed(a,b,m,N_actions,grid,action_dims)
    polynomials = []
    for i in 1:N_actions
        push!(polynomials, init_Q_continuous(a,b,m,action_dims))
    end 
    return Q_mixed(collect(1:N_actions), grid, action_dims, polynomials)
end
        

        
function evaluate_Q_continuous(Q,s,a)
    return OLS_chebyshev.evaluate_polynomial(vcat(a,s),Q.polynomial)
end
        
function evaluate_Q_continuous(Q,s,a_disc, a_cont)
    return OLS_chebyshev.evaluate_polynomial(vcat(a_cont,s),Q.polynomials[a_disc])
end


        
# updating methods       
function update_Q_discrete!(Q,a,s,Q_data)
    for i in Q.A
        inds = a .== i
        s_ = s[inds,:]
        Q_ = Q_data[inds]
        OLS_chebyshev.update_polynomial!(Q.polynomials[i],Q_,s_)
    end
end 
        
function update_Q_continuous!(Q,a,s,Q_data)
    OLS_chebyshev.update_polynomial!(Q.polynomial,Q_,hact(a,s))    
end
        
function evaluate_Q_mixed(Q,s,a_disc, a_cont)
    for i in Q.A
        inds = a_disc .== i
        a_ = a_cont[inds]
        s_ = s[inds,:]
        Q_ = Q_data[inds]
        OLS_chebyshev.update_polynomial!(Q.polynomials[i],Q_,hcat(a_,s_))
    end
end        

            


end # module 