module MvGaussHermite

using FastGaussQuadrature 
using LinearAlgebra 

export quadrature, mutableQuadrature

struct quadrature
    weights::AbstractVector{Float64}
    nodes::AbstractVector{AbstractVector{Float64}}
    Cov::AbstractMatrix{Float64} # covariance matrix
    dims::Int64 # number of dimensions 
    m::Int64 # order of aproximation 
    n::Int64
end 


mutable struct mutableQuadrature
    weights::AbstractVector{Float64} # quadrature weights 
    nodes::AbstractVector{AbstractVector{Float64}} # beleif state nodes
    Cov::AbstractMatrix{Float64} # covariance matrix beleif state
    dims::Int64 # number of dimensions states
    m::Int64 # order of aproximation 
    n::Int64 # number of nodes
end 

# I need to check the signs of the sin terms here
# it works in R2 and R3 but not in R4
function planar_rotation(d,theta)
    R = 1.0*Matrix(I,d,d)
    for i in 1:(d-1)
        R_ = 1.0*Matrix(I,d,d)
        R_[i,i] = cos(theta)
        R_[i,i+1] = -sin(theta)
        R_[i+1,i] = sin(theta)
        R_[i+1,i+1] = cos(theta)
        R .= R*R_
    end 
    return R
end 


"""
    nodes_grid(nodes, weights, dims)

makes a grid of Gauss hermite nodes and weights. I'm not quite sure how to write this
function and for and number of dims and above three MC is probably better any ways 
so I have defined sperate functions for dims in {1,2,3}

"""
function nodes_grid(nodes, weights, dims)
    @assert dims in [1,2,3,4]
    
    if dims == 1
        return nodes, weights
    elseif dims ==2
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^2)
        weights_vec = zeros(n^2)
        acc = 0
        for i in 1:n
            for j in 1:n
                acc += 1
                nodes_vec[acc] .= [nodes[i], nodes[j]]
                weights_vec[acc] = weights[i]*weights[j]
            end
        end 
        return nodes_vec, weights_vec
    elseif dims == 3
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^dims)
        weights_vec = zeros(n^3)
        acc = 0
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    acc += 1
                    nodes_vec[acc] .= [nodes[i], nodes[j], nodes[k]]
                    weights_vec[acc] = weights[i]*weights[j]*weights[k]
                end
            end
        end 
        return nodes_vec, weights_vec
    elseif dims == 4
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^dims)
        weights_vec = zeros(n^4)
        acc = 0
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    for l in 1:n
                        acc += 1
                        nodes_vec[acc] .= [nodes[i], nodes[j], nodes[k], nodes[l]]
                        weights_vec[acc] = weights[i]*weights[j]*weights[k]*weights[l]
                    end
                end
            end
        end 
        return nodes_vec, weights_vec
    end 
    
end 



function init(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    nodes, weights = nodes_grid(nodes, weights, dims)
    
    nodes = broadcast(x -> broadcast(v -> v, x), nodes)


    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
   
    R = planar_rotation(dims,pi/4)
 
    # transform and plot 
    nodes = broadcast(x -> (S*rV)*R*x.+mu, nodes)
    
    return quadrature(weights, nodes, Cov, dims, m, length(nodes))
    
end 




function init(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64},theta::Float64)
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    nodes, weights = nodes_grid(nodes, weights, dims)
    
    nodes = broadcast(x -> broadcast(v -> v, x), nodes)


    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    
    R = planar_rotation(dims,pi/4)
 
    # transform and plot 
    nodes = broadcast(x -> S*rV*R*x.+mu, nodes)
    nodes = nodes[weights .> theta]
    weights = weights[weights .> theta]
    
    return quadrature(weights, nodes, Cov, dims, m, length(nodes))
    
end 



function init_mutable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    nodes, weights = nodes_grid(nodes, weights, dims)
    
    nodes = broadcast(x -> [x[1],x[2]], nodes)


    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,2,2).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(dims,pi/4)

    # transform and plot 
    nodes = broadcast(x -> S*rV*R*x.+mu, nodes)
    
    return mutableQuadrature(weights, nodes, Cov, dims, m, length(nodes))
    
end 


function init_mutable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64},theta::Float64)
    dims = size(Cov)[1]
    # get weights
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    # convert weights from exp(-x^2) to (pi/2)^1/2*exp(-1/2x^2)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    nodes, weights = nodes_grid(nodes, weights, dims)
    
    nodes = broadcast(x -> [x[1],x[2]], nodes)

    
    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,2,2).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(dims,pi/4)

    # transform and plot 
    nodes = broadcast(x -> S*rV*R*x .+ mu, nodes)
    nodes = nodes[weights .> theta]
    weights = weights[weights .> theta]
    
    return mutableQuadrature(weights, nodes, Cov, dims, m, length(nodes))
    
end 

"""
Updates the transformation of the nodes with a new covariance matrix 
"""
function update!(mutableQuadrature, mu::AbstractVector{Float64}, Cov::AbstractMatrix{Float64})

    
    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,2,2).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(mutableQuadrature.dims,pi/4)

    # transform and plot 
    mutableQuadrature.nodes = broadcast(x -> S*rV*R*x .+ mu, mutableQuadrature.nodes)
    mutableQuadrature.Cov = Cov
end 

"""
specific update function for POMDPs where the covariance matrix is block
diaganol with a block for the observaiton noise covaraicne Cov_y whcih 
does not change and a block for the state uncertianty Cov_x which will
here only the Cov_x block is given. I assume that the Cov_x block is 
the upper left hand corner of the matrix. 
"""
function update_bellman!(mutableQuadrature, x_cov)
    dims_x = size(x_cov)
    mutableQuadrature.Cov[1:dims_x,1:dims_x] .= x_cov
    
    # spectral decomposition
    estuff = eigen(mutableQuadrature.Cov)
    rV = sqrt.(1.0*Matrix(I,2,2).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(mutableQuadrature.dims,pi/4)

    # transform and plot 
    mutableQuadrature.nodes = broadcast(x -> S*rV*R*x, mutableQuadrature.nodes)
    
end 
"""
    expected_value(f::Function, quadrature::quadrature,  mu::AbstractVector{Float64})

computes the expecctation of a gausian random variable with mean mu and covariance quadrature.Cov
with respect to a function f
"""
function expected_value(f::Function, quadrature)#
    return sum(f.(broadcast(x -> x, quadrature.nodes)).*quadrature.weights)#/sqrt(pi^quadrature.dims)
end 

function expected_value(f::Function, quadrature)#::mutableQuadrature
    return sum(f.(broadcast(x -> x, quadrature.nodes)).*quadrature.weights)#/sqrt(pi^quadrature.dims)
end 



end # module 