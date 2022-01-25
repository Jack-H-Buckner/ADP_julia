"""
    BellmanOpperators

Provides methods to evaluate the Bellman equation for each of the POMDP objects. 

The primay function defined in this module is `bellman` which evaluates the bellman opperator give
a POMDP object, and initial state and a value function object. If an action variable `a` is provided 
then the algorithm will calcualte the expected value of taking that action in the current state. If
no action is provided then the function will solve for the best action using the `Optim.jl` package. 

There are two primary computational challenges related to solving the bellman opperator: optimization and 
integration. Seperate methos are defined to apply the most aproporate method depending on the input arguments.

For `POMDP_particleFilter` objects Montecarlo integration is used, and Gauss Hermite quadrature is used for 
`POMDP_kalmanFilter` objects. The implementation of the Gausian quadrature depends on weather or not the observation
model is linear. If it is linear then 

The optimizaiton method is chosen based on the Type of the POMDP.actions object. Nelder Mead is used for 
multivariate continuous aciton, Golden search for univariate continuous, and brute force is used for 
discrete action spaces. 
"""
module BellmanOpperators

#using Optim
#using Roots
using KalmanFilters
include("MvGaussHermite.jl")
include("utils.jl")

"""
    propogate_observation_model(x_hat,Cov,H)

propogates state uncertianty through observation model if H is a function
the gassian quadrature is used if it is a matrix then a linear transform
is used. 

x_hat - mean estimate for x
Cov - covariance of beleif state
H - observation model, matrx of function 
"""
function propogate_observation_model(x_hat::AbstractVector{Float64},x_cov::AbstractMatrix{Float64},
                                     H::Function,Quad)
    MvGaussHermite.update!(Quad, x_hat, x_cov)
    vals = H.(Quad.nodes)
    y_hat = sum(vals.*Quad.weights)
    y_cov = sum(broadcast(v-> (v.-y_hat)*transpose(v.-y_hat), vals).* Quad.weights)
    return y_hat, y_cov
end 


function propogate_observation_model(x_hat::AbstractVector{Float64},x_cov::AbstractMatrix{Float64},
                                     H::AbstractMatrix{Float64})
    y_hat = H * x_hat
    y_cov = H * x_cov * transpose(H)
    return y_hat, y_cov
end 


"""
    new_state(y, x_hat, x_cov,H,Sigma_O)

computes measurement update using KalmanFilters.jl 
"""
function new_state(y::AbstractVector{Float64}, x_hat::AbstractVector{Float64},
                    x_cov::AbstractMatrix{Float64},H::AbstractMatrix{Float64},
                    Sigma_O::AbstractMatrix{Float64})
    
    mu = KalmanFilters.measurement_update(x_hat, x_cov,y,H,Sigma_O)
    x_hat, x_cov = KalmanFilters.get_state(mu), KalmanFilters.get_covariance(mu)
    
    return x_hat, x_cov
end 


"""
    new_state(y, x_hat, x_cov,H,Sigma_O)

computes measurement update using KalmanFilters.jl 
"""
function new_state(y::AbstractVector{Float64}, x_hat::AbstractVector{Float64},
                    x_cov::AbstractMatrix{Float64},H::Function,
                    Sigma_O::AbstractMatrix{Float64})
    
    mu = KalmanFilters.measurement_update(x_hat, x_cov,y,H,Sigma_O)
    x_hat, x_cov = KalmanFilters.get_state(mu), KalmanFilters.get_covariance(mu)
    
    return x_hat, x_cov
end 

"""
    reshape_state(x_hat, x_cov)

reshapes the beleif state represented by a mean vector and covariance matrix to a 
vector of length n + n(n+1)/2
"""
function reshape_state(x_hat::AbstractVector{Float64}, x_cov::AbstractMatrix{Float64})
    n = length(x_hat)
    v = zeros(floor(Int,n + n*(n+1)/2))
    v[1:n] .= x_hat
    k = n
    for i in 1:n
        for j in 1:n
            if j <= i
                k += 1
                v[k] = x_cov[i,j]
            end 
        end
    end 
    return v
end 


"""
    reshape_state(B)

Maps a vector to a mean and covariance 
"""
function reshape_state(B::AbstractVector{Float64})
    d = floor(Int, -3/2+sqrt(9/4+2*length(B)))
    x_hat = B[1:d]
    x_cov = zeros(d,d)
    k = d
    for i in 1:d
        for j in 1:d
            if j == i
                k += 1
                x_cov[i,j] = B[k] 
            elseif j < i
                k += 1
                x_cov[i,j],x_cov[j,i] = B[k],B[k]  
            end 
        end
    end 
    return x_hat, x_cov
end 



"""
    integrate_bellman(x_hat, x_cov, a, V, POMDP)

s - state vector length n + n*(n+1)/2
a - action taken
V - value function (takes state )
POMDP - 
"""
function expectation(s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            V::Function, POMDP, Quad_x, Quad_y)
    
    # convert state vector to mean and covariance 
    x_hat, x_cov = reshape_state(s)
    
    ### time update 
    tu = KalmanFilters.time_update(x_hat,x_cov, x ->POMDP.T(x,a),  POMDP.Sigma_N)
    x_hat, x_cov = KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
    
    #observaton quadrature nodes 
    H = x -> POMDP.H(x,a) # define measurment function 
    y_hat, y_cov = propogate_observation_model(x_hat, x_cov,H,Quad_x)
    
    y_cov .+= POMDP.Sigma_O
    MvGaussHermite.update!(Quad_y,y_hat,y_cov)
    
    
    ### new states 
    new_states = broadcast(y -> new_state(y, x_hat,x_cov, H, POMDP.Sigma_O), Quad_y.nodes)
    new_states = broadcast(x -> reshape_state(x[1],x[2]),new_states)
    
    vals = broadcast(x -> V(x, a), new_states)
    return sum(vals .* Quad_y.weights)
end 

end # module 