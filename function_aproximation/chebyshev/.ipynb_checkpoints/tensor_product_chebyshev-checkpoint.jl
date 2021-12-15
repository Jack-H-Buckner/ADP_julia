module TP_chebyshev

include("utils.jl")
    
    
# define a data structure to save 
# informaiton for interpolation
mutable struct interpolation
    d::Int64 # dimensions
    a::AbstractVector{Float64} # lower bounds
    b::AbstractVector{Float64} # upper bounds
    m::Int # nodes per dimension
    nodes::AbstractMatrix{Float64} # m by d matrix with nodes in each dimension
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
    nodes = (z.+1).*transpose((b.-a)./2 .- a)
    # initialize values as zero
    values = zeros(ntuple(x -> m, d))
    coefs = zeros(binomial(m+d,m))
    alpha = utils.collect_alpha(m,d)
    return interpolation(d,a,b,m,nodes,values,coefs,alpha)
end 


# define a function to update the values and 
# coeficents for the interpolation
function update_interpolation(current, new_values)
    # check size of new values
    @assert length(new_values) == current.m^current.d
    @assert all(size(new_values) .== current.m)
    # update values
    current.values = new_values
    # compute coeficents
    m = current.m
    d = current.d
    alpha = current.alpha
    coefs = utils.compute_coefs(d,m,new_values,alpha)
    current.coeficents = coefs
    return current
end 

    
# evaluluates the interpolation at all points inclueded x
function evaluate_interpolation(x,interpolation)
    a = interpolation.a
    b = interpolation.b   
    z = broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,interpolation.alpha, interpolation.coeficents),z)
    return v
end 


end #modle