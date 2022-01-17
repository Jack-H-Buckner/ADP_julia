"""
This file contains code to implement the unscented Kalmna Filter algorithm
with POMDP objects. The code will be based around the methods defined in 
the publicly avaialbe KalmanFilters.jl but with wrapers so it plays nicely 
with the POMDP modle objects. It will also include methods to handel problems
with discrete hidden states. 
"""
module KalmanFilters
using LinearAlgebra


mutable struct unscentedFilter
    x::AbstractVector{Float64} # mean estimate
    P::AbstractMatrix{Float64} # covariance estimate
    X::AbstractVector{AbstractVector{Float64}} # sigma points
    Wm::AbstractVector{Float64} # mean weights
    Wc::AbstractVector{Float64} # cov weights
    L::Int64
    lambda::Float64 # tuning paramters
    beta::Float64 # tuning parameters
    alpha::Float64 # tuning parameters
    kappa::Float64
end 

function calc_weights(L, alpha, beta, kappa)
    lambda = alpha^2*(L-kappa)-L
    
    # set weights
    Wm,Wc,X = zeros(2*L+1),zeros(2*L+1),broadcast(i -> zeros(L), 1:(2*L+1))
    Wm[1] = lambda/(L+lambda) 
    Wc[1] = lambda/(L+lambda) + (1-alpha^2+beta)
    Wm[2:end] .= 1/(2*(L+lambda))
    Wc[2:end] .= 1/(2*(L+lambda))
    return lambda, Wm, Wc, X
end 

function init(x0::AbstractVector{Float64},P0::AbstractMatrix{Float64},alpha::Float64,beta::Float64,kappa::Float64)
    # calcualte paramters
    L = length(x0)
    lambda,Wm,Wc,X = calcWeights(L, alpha, beta, kappa)
    return unscentedFilter(x0,P0,X,Wm,Wc,L,lambda,beta,alpha,kappa)
end

function init(x0::AbstractVector{Float64},P0::AbstractMatrix{Float64},alpha::Float64)
    # set defaults
    beta = 2.0
    kappa = 0.0
    # calcualte paramters
    L = length(x0)
    lambda,Wm,Wc,X = calcWeights(L, alpha, beta, kappa)
    return unscentedFilter(x0,P0,X,Wm,Wc,L,lambda,beta,alpha,kappa)
end


function init(x0::AbstractVector{Float64},P0::AbstractVector{Float64},alpha::Float64)
    # set defaults
    beta = 2.0
    kappa = 0.0
    # calcualte paramters
    L = length(x0)
    lambda,Wm,Wc,X = calcWeights(L, alpha, beta, kappa)
    return unscentedFilter(x0,P0,X,Wm,Wc,L,lambda,beta,alpha,kappa)
end



function sigma_points!(unscentedFilter)
    rootP = Array(LinearAlgebra.cholesky(sqrt((unscentedFilter.L+unscentedFilter.lambda))*unscentedFilter.P))
    for i in 1:(2*unscentedFilter.L+1)
        if i == 1
            unscentedFilter.X[i] = unscentedFilter.x
        elseif i < unscentedFilter.L+2
            unscentedFilter.X[i] = unscentedFilter.x .+ rootP[i-1,:]
        else
            unscentedFilter.X[i] = unscentedFilter.x .- rootP[i-unscentedFilter.L-1,:]
        end 
    end 
end 



function sum_vec(x)
    d = length(x[1])
    n = length(x)
    acc = zeros(d)
    for i in 1:n
        acc .+= x[i]
    end
    return acc
end


function sum_mat(x)
    cols, rows = size(x[1])
    n = length(x)
    acc = zeros(cols, rows)
    for i in 1:n
        acc .+= x[i]
    end
    return acc
end

### d > 1  
"""
    timeUpdate!(unscentedFilter, F, H, a)

T - state transition function
N - process noise covariance
a - additional parameters for T, actions 
"""
function time_update!(unscentedFilter, T::Function, N::AbstractMatrix{Float64}, a::AbstractVector{Float64})
    sigma_points!(unscentedFilter)
    Fx = broadcast(x -> T(x,a), unscentedFilter.X)
    unscentedFilter.x = sum_vec(Fx .* unscentedFilter.Wm)
    unscentedFilter.P = sum_mat(broadcast(v-> (v .- unscentedFilter.x)*transpose(v .- unscentedFilter.x), Fx).* unscentedFilter.Wc)
    unscentedFilter.P .+= N
end 




"""
    bayesUpdate!(unscentedFilter, F, H, a)

yt - observation
H - observation model (function of action a)
R - observation noise covariance 
a - 
"""
function bayes_update!(unscentedFilter, yt::AbstractVector{Float64}, H::AbstractMatrix{Float64}, 
                      R::AbstractMatrix{Float64}, a::AbstractVector{Float64})
    zt = yt .- H*unscentedFilter.x
    St = H*unscentedFilter.P*transpose(H) .+ R 
    Kt = unscentedFilter.P*transpose(H)*inv(St)
    unscentedFilter.x .+= reshape(Kt.*zt, (unscentedFilter.L))
    unscentedFilter.P = (Matrix(I,unscentedFilter.L,unscentedFilter.L) .- Kt*H)*unscentedFilter.P
end


function bayes_update!(unscentedFilter, yt::AbstractVector{Float64}, H::AbstractMatrix{Float64}, 
                      R::AbstractVector{Float64}, a::AbstractVector{Float64})
    zt = yt .- H*unscentedFilter.x
    St = H*unscentedFilter.P*transpose(H) .+ R 
    Kt = unscentedFilter.P*transpose(H)./St
    unscentedFilter.x .+= reshape(Kt.*zt, (unscentedFilter.L))
    unscentedFilter.P = Matrix(Hermitian((Matrix(I,unscentedFilter.L,unscentedFilter.L) .- Kt*H)*unscentedFilter.P))
end



# d = 1 
function time_update!(unscentedFilter, T::Function, N::AbstractVector{Float64}, a::AbstractVector{Float64})
    Fx = broadcast(x -> T(x,a), unscentedFilter.X)
    unscentedFilter.x = sum_vec(Fx .* unscentedFilter.Wm)
    unscentedFilter.P = sum_vec(broadcast(v-> (v .- unscentedFilter.x)*(v .- unscentedFilter.x), Fx).* unscentedFilter.Wc)
    unscentedFilter.P .+= N
end 


function bayes_update!(unscentedFilter, zt::AbstractVector{Float64}, H::AbstractVector{Float64}, 
                      R::AbstractVector{Float64}, a::AbstractVector{Float64})
    yt = zt .- H*unscentedFilter.x
    St = H*unscentedFilter.P*H .+ R 
    Kt = unscentedFilter.P*H*inv(St)
    unscentedFilter.x .+= Kt*yt
    unscentedFilter.P = ([1.0] .- Kt*H)*unscentedFilter.P
end

end # module