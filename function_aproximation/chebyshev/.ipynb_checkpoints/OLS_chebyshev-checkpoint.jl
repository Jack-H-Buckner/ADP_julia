"""
OLS_chebyshev

data structures and assocaited methods 
"""
module OLS_chebyshev

include("utils.jl")

mutable struct polynomial
    d::Int # number of dimensions
    a::AbstractVector{Float64} # lower bounds 
    b::AbstractVector{Float64} # upper bounds 
    N::Int # order of polynomial 
    alpha::AbstractVector{Any} # order in each dimension for each term 
    coeficents::AbstractVector{Float64} # polynomial coeficents
    extrapolate
end 


function init_polynomial(a,b,N,extrapolate)
    @assert length(a) == length(b)
    d = length(a)
    coeficents = zeros(binomial(N+d,d))
    alpha = utils.collect_alpha(N,d)
    P = polynomial(d,a,b,N,alpha,coeficents,extrapolate)
    return P
end 


# y - vector of floats
# x - d by nx matrix of floats 
function update_polynomial!(polynomial,y,x)
    @assert size(x)[1] > binomial(polynomial.N+polynomial.d,polynomial.d)
    X = utils.regression_matrix(x, polynomial)
    coefs = (transpose(X)*X)\transpose(X)*y
    polynomial.coeficents = reshape(coefs,binomial(polynomial.N+polynomial.d,polynomial.d))
    #return polynomial
end 


function bound!(x, polynomial)
    x_low = x .< polynomial.a
    x_high = x .> polynomial.b
    x[x_low] = polynomial.a[x_low]
    x[x_high] = polynomial.b[x_high]
    return x
end 

function bound_error!(x, polynomial)
    x_low = x .< a
    x_high = x .> b
    return any(x_low)|any(x_high)
end 


# evaluluates the interpolation at all points inclueded x
function evaluate_polynomial(x,polynomial)
    a = polynomial.a
    b = polynomial.b 
    if polynomial.extrapolate
        x = broadcast(v -> bound!(v,polynomial), x)
    else
        @assert all(broadcast(v -> bound_error!(v,polynomial), x))
    end 
    z = broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,polynomial.alpha, polynomial.coeficents),z)
    return v
end 

end # module 