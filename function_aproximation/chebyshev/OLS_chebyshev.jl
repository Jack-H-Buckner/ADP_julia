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
end 


function init_polynomial(a,b,N)
    @assert length(a) == length(b)
    d = length(a)
    coeficents = zeros(binomial(N+d,d))
    alpha = utils.collect_alpha(N,d)
    P = polynomial(d,a,b,N,alpha,coeficents)
    return P
end 


# y - vector of floats
# x - d by nx matrix of floats 
function update_polynomial(y,x,polynomial)
    @assert size(x)[1] > binomial(polynomial.N+polynomial.d,polynomial.d)
    X = utils.regression_matrix(x, polynomial)
    coefs = (transpose(X)*X)\transpose(X)*y
    polynomial.coeficents = reshape(coefs,binomial(polynomial.N+polynomial.d,polynomial.d))
    return polynomial
end 

# evaluluates the interpolation at all points inclueded x
function evaluate_polynomial(x,polynomial)
    a = polynomial.a
    b = polynomial.b   
    z = broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,polynomial.alpha, polynomial.coeficents),z)
    return v
end 

end # module 