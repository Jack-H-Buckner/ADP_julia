module ValueFunctions

include("utils.jl")

#############################
### chebyshev polynomials ###
#############################

# grid interpolation

# define a data structure to save 
# informaiton for interpolation
mutable struct chebyshevInterpolation
    d::Int64 # dimensions
    a::AbstractVector{Float64} # lower bounds
    b::AbstractVector{Float64} # upper bounds
    m::Int # nodes per dimension
    nodes::AbstractMatrix{Float64} # m by d matrix with nodes in each dimension
    grid::AbstractArray{}
    values # value associated with each node (m^d entries)
    coeficents::AbstractVector{Float64} # coeficents for computing the polynomial
    alpha::AbstractVector{Any}  # vector of tuples with m integer arguments 
end  

# define a function to initialize the 
# interpolation data structure with zeros
function init_interpolation(a,b,m)
    @assert length(a) == length(b)
    d = length(a)
    # calcualte nodes
    f = n -> -cos((2*n-1)*pi/(2*m))
    z = f.(1:m)
    

    nodes = (z.+1)./2 .*transpose(b.-a) #.- a
    
    nodes = mapslices(x-> x .+ a, nodes, dims = 2)

    grid = utils.collect_nodes(nodes) 
    # initialize values as zero
    values = zeros(ntuple(x -> m, d))
    coefs = zeros(binomial(m+d,m))
    alpha = utils.collect_alpha(m,d)
    return chebyshevInterpolation(d,a,b,m,nodes,grid,values,coefs,alpha)
end 


# define a function to update the values and 
# coeficents for the interpolation
function update_interpolation!(interpolation, new_values)
    # check size of new values
    @assert length(new_values) == interpolation.m^interpolation.d
    @assert all(size(new_values) .== interpolation.m)
    # update values
    interpolation.values = new_values
    # compute coeficents
    m = interpolation.m
    d = interpolation.d
    alpha = interpolation.alpha
    coefs = utils.compute_coefs(d,m,new_values,alpha)
    interpolation.coeficents = coefs
    #return current
end 

    
# evaluluates the interpolation at all points inclueded x
function evaluate_interpolation(x,interpolation)
    a = interpolation.a
    b = interpolation.b   
    z = broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,interpolation.alpha, interpolation.coeficents),z)
    return v
end 


"""
define method to call 
"""
function (p::chebyshevInterpolation)(x)
    a = p.a
    b = p.b   
    
    # scale x values to (-1,1)
    z = (x.-a).*2 ./(b.-a) .- 1 #broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    
    # extrapolation
    z[z .> 1.0] .= 0.9999999
    z[z .< -1.0] .= -0.9999999
    v = utils.T_alpha(z,p.alpha, p.coeficents) #broadcast(x -> utils.T_alpha(x,p.alpha, p.coeficents),z)
    return v
end 


# regression model for ADP


mutable struct chebyshevPolynomial
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

function bound_error(x, polynomial)
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
        @assert all(broadcast(v -> bound_error(v,polynomial), x))
    end 
    z = broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,polynomial.alpha, polynomial.coeficents),z)
    return v
end 


function (p::chebyshevPolynomial)(x)
    a = p.a
    b = p.b 
    if p.extrapolate
        x = broadcast(v -> bound!(v,p), x)
    else
        @assert all(broadcast(v -> bound_error(v,p), x))
    end 
    z = broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,p.alpha, p.coeficents),z)
    return v
end 
        
        

#############################
###     other methods     ###
#############################  
    
end 