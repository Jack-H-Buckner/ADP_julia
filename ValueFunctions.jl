module ValueFunctions

include("utils.jl")

#############################
### chebyshev polynomials ###
#############################

# grid interpolation

# define a data structure to save 
# informaiton for interpolation
mutable struct chebyshevInterpolation{T}
    d::Int64 # dimensions
    a::AbstractVector{Float64} # lower bounds
    b::AbstractVector{Float64} # upper bounds
    m::Int # nodes per dimension
    nodes::AbstractMatrix{Float64} # m by d matrix with nodes in each dimension
    grid::AbstractArray{AbstractVector{Float64}}
    values::AbstractVector{Float64} # value associated with each node (m^d entries)
    coeficents::AbstractVector{Float64} # coeficents for computing the polynomial
    alpha::AbstractVector{NTuple{T,Int64}}  # vector of tuples with m integer arguments 
    v::AbstractVector{AbstractVector{Float64}} # stores intermidiate values 
    T_alpha_i_grid::AbstractVector{AbstractVector{Float64}}
    vT_sum::AbstractVector{Float64}
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

    z = (z.+1)./2 .*transpose(repeat([2.0],d))
    z = mapslices(x-> x .-  1.0, z, dims = 2)
    grid = utils.collect_nodes(nodes) # nodes on desiered domain
    grid_mapped = utils.collect_nodes(z) # nodes mapped to -1,1
    # initialize values as zero
    values = 1.0*zeros(m^d) #
    
    coefs = zeros(binomial(m+d,m))
    alpha = utils.collect_alpha(m,d)
    v = broadcast(i -> 1.0*zeros(length(alpha)), 1:Threads.nthreads())#zeros(ntuple(x -> m, d))
    T_alpha_i_grid = broadcast(alpha_i -> broadcast(x -> utils.T_alpha_i(alpha_i,x), grid_mapped),alpha)
    vT_sum = 1.0*zeros(length(alpha))
    return chebyshevInterpolation{d}(d,a,b,m,nodes,grid,values,coefs,alpha, v,T_alpha_i_grid,vT_sum)
end 


# define a function to update the values and 
# coeficents for the interpolation
function update_interpolation!(interpolation, new_values)
    # check size of new values
    #@assert length(new_values) == interpolation.m^interpolation.d
    @assert all(size(new_values) .== interpolation.m^interpolation.d)
    # update values
    interpolation.values = new_values
    # compute coeficents
    m = interpolation.m
    d = interpolation.d
    alpha = interpolation.alpha
    #coefs = utils.compute_coefs(d,m,new_values,alpha,interpolation.T_alpha_i_grid)
    coefs = utils.compute_coefs1(d,m,new_values,alpha,interpolation.T_alpha_i_grid,interpolation.vT_sum)
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
    
    # scale x values to (-1,1)
    z = (x.-p.a).*2 ./(p.b.-p.a) .- 1 #broadcast(x -> (x.-a).*2 ./(b.-a) .- 1,x)
    
    # extrapolation
    z[z .> 1.0] .= 0.9999999
    z[z .< -1.0] .= -0.9999999
   
    # note that problem arise when intermidiates are used while multi threading 

    v = utils.T_alpha!(p.v[Threads.threadid()],z,p.alpha, p.coeficents) 
    #v = utils.T_alpha!(zeros(length(p.v)),z,p.alpha, p.coeficents) 
    #broadcast(x -> utils.T_alpha(x,p.alpha, p.coeficents),z)
    #v = utils.T_alpha!(p.v,z,p.alpha, p.coeficents) #broadcast(x -> utils.T_alpha(x,p.alpha, p.coeficents),z)
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

    x = broadcast(v -> bound!(v,p), x)

    z = broadcast(x -> (x.-p.a).*2 ./(p.b.-p.a) .- 1,x)
    v = broadcast(x -> utils.T_alpha(x,p.alpha, p.coeficents),z)
    return v
end 
        
        


    
#############################
###     other methods     ###
#############################  
    
# define a data structure to save 
# informaiton for interpolation
mutable struct guasianBeleifsInterp2d

    lower_mu::AbstractVector{Float64} # lower bounds
    upper_mu::AbstractVector{Float64} # upper bounds
    
    upper_sigma::AbstractVector{Float64}
    
    covBound::Float64
    m::Int64
    nodes::AbstractArray{Tuple{AbstractVector{Float64}, AbstractMatrix{Float64}}}
    chebyshevInterpolation
end  

    
    
function inv_map_node!(mu::AbstractVector{Float64}, cov::AbstractMatrix{Float64}, z, lower_mu, upper_mu, upper_sigma, covBound)
    mu .= (z[1:2].+1) .* (upper_mu.-lower_mu)./2.0 .+lower_mu
    cov[1,1] = (z[3].+1)* (upper_sigma[1])/2.0 
    cov[2,2] = (z[4].+1)* (upper_sigma[2])/2.0 
    covBound = sqrt(cov[1,1]*cov[2,2]) - 0.000001
    cov[1,2] = covBound*(z[5]+1)/2.0 
    cov[2,1] = covBound*(z[5]+1)/2.0
end    

"""
    init_guasianBeleifsInterp2d(m, lower_mu, upper_mu)

Initializes the interpolation for a POMDP with 2d gausian beleif state
"""
function init_guasianBeleifsInterp2d(m, lower_mu, upper_mu)
    @assert m < 15
    range = upper_mu .- lower_mu
    upper_sigma = range./3
    
    covBound = sqrt(upper_sigma[1]*upper_sigma[2])
    a = repeat([-1.0],5)
    b = repeat([1.0],5)
    interp = init_interpolation(a,b,m)
    
    nodes = broadcast( i-> (zeros(2),zeros(2,2)),1:m^5)
    
    broadcast(i -> inv_map_node!(nodes[i][1],nodes[i][2], interp.grid[i],lower_mu, upper_mu, upper_sigma, covBound), 1:m^5)
    
    guasianBeleifsInterp2d(lower_mu,upper_mu,upper_sigma,covBound,m,nodes,interp)
end 

"""
maps a mean and covariance matrix to (-1,1)^5
    
Over writes the state vector supplied 
"""
function map_node!(z::AbstractVector{Float64}, mu::AbstractVector{Float64}, cov::AbstractMatrix{Float64}, Binterp)
    z[1:2] = (mu.-Binterp.lower_mu).*2 ./(Binterp.upper_mu.-Binterp.lower_mu) .- 1
    z[3] = (cov[1,1] )*2 / (Binterp.upper_sigma[1]) - 1
    z[4] = (cov[2,2] )*2 / (Binterp.upper_sigma[2]) - 1
    Binterp.covBound = sqrt(cov[1,1]*cov[2,2])*(1-10^-8)
    z[5] = cov[1,2] /Binterp.covBound
    z[5] = 2.0*z[5] - 1.0
end 
    
"""
maps a point in (-1,1)^5 to a mean and covariance matrix
    
Over writes the mean and covariance supplied 
"""
function inv_map_node!(mu::AbstractVector{Float64}, cov::AbstractMatrix{Float64}, z::AbstractVector{Float64}, covBound)
    mu .= (z[1:2].+1) .* (Binterp.upper_mu.-Binterp.lower_mu)./2.0 .+ Binterp.lower_mu
    cov[1,1] = (z[3].+1)* (Binterp.upper_sigma[1])/2.0 
    cov[2,2] = (z[4].+1)* (Binterp.upper_sigma[2])/2.0 
    covBound = sqrt(cov[1,1]*cov[2,2]) - 0.000001
    cov[1,2] = covBound*(z[5]+1)/2.0 
    cov[2,1] = covBound*(z[5]+1)/2.0 
end 
        
function (p!::guasianBeleifsInterp2d)(z::AbstractVector{Float64},s::Tuple{AbstractVector{Float64},AbstractMatrix{Float64}})

    map_node!(z, s[1], s[2], p!)
    
    # extrapolation
    z[z .> 1.0] .= 0.9999999
    z[z .< -1.0] .= -0.9999999
    #p!.chebyshevInterpolation.v
    v = utils.T_alpha!(p!.chebyshevInterpolation.v[Threads.threadid()],z,p!.chebyshevInterpolation.alpha, p!.chebyshevInterpolation.coeficents) 
    return v
end
    
    
"""
    update_guasianBeleifsInterp2d!(V,vals)

V - guasianBeleifsInterp2d object
vals - vector of new values with same length as nodes in interpolation 
"""
function update_guasianBeleifsInterp2d!(V,vals::AbstractVector{Float64})
    update_interpolation!(V.chebyshevInterpolation, vals)
end 
    

"""
    adjGausianBeleifsInterp

A value funciton object that is the sum of two parts. 
The function maps a mean and covariance function 
baseValue is a functon that 
"""
mutable struct adjGausianBeleifsInterp
    baseValue::chebyshevInterpolation
    uncertantyAdjustment::guasianBeleifsInterp2d
    m1::Int64
    m2::Int64
end 
    
    
function init_adjGausianBeleifsInterp(m1, m2, lower_mu, upper_mu)
    baseValue = init_interpolation(lower_mu, upper_mu, m1)
    uncertantyAdjustment = init_guasianBeleifsInterp2d(m2, lower_mu, upper_mu)
    adjGausianBeleifsInterp(baseValue,uncertantyAdjustment,m1,m2)
end 
    
function (V!::adjGausianBeleifsInterp)(z,s)
    return V!.baseValue(s[1]) + V!.uncertantyAdjustment(z,s)
end 
    

"""
    update_base()

V - adjGausianBeleifsInterp
values - vector of length V.m1^2 
"""
function update_base!(V,vals)
    update_interpolation!(V.baseValue, vals)
end

"""
    update_adjustment(V, vals)


"""
function update_adjustment!(V, vals)
    vals_up = vals .- broadcast(i -> V.baseValue(V.uncertantyAdjustment.nodes[i][1]),
            1:length(V.uncertantyAdjustment.nodes))
    update_guasianBeleifsInterp2d!(V.uncertantyAdjustment,vals_up)
end

end 