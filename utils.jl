"""
useful funcitons
"""
module utils
phi(y) = exp(-1/2*y^2)/sqrt(2*pi)

function hermite(x,degree)
    if degree == 4
        return x^4 - 6x^2 + 3
    elseif degree == 3
        return x^3 - 3*x
    end 
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

"""
computes cumulats given  weighted sample 
"""
function cumulant(sample, weights, degree)
    @assert degree < 5
    if degree == 1
        return sum(sample .* weights)
    elseif degree ==2
        mu = sum(sample .* weights)
        return sum((sample .- mu).^2 .* weights )
    elseif degree ==3
        mu = sum(sample .* weights)
        return sum((sample .- mu).^3 .* weights )
    elseif degree ==4
        mu = sum(sample .* weights)
        return sum((sample .- mu).^4 .* weights ) - 3*cumulant(sample, weights, 2)^2
    end 
end 



### tools for chebyshev polynomials



# the nth degree chebyshev polynomial
# evaluated at x
function T(x::Float64,n::Int64)
    return cos(n*acos(x))
end 

    

# the product of chebyshev polynomials
function T_alpha_i( alpha_i::NTuple{2,Int64},x::AbstractVector{Float64})
    #return prod(T.(x,alpha_i)) 
    v = 1.0
    for i in 1:length(x)
        v *= cos(alpha_i[i]*acos(x[i]))
    end 
    return v
end
        
function T_alpha_i( alpha_i::NTuple{5,Int64},x::AbstractVector{Float64})
    #return prod(T.(x,alpha_i)) 
    v = 1.0
    for i in 1:length(x)
        v *= cos(alpha_i[i]*acos(x[i]))
    end 
    return v
end   

            
                    
# the product of chebyshev polynomials
# summed for each value of alpha
function T_alpha!(v::AbstractVector{Float64},x::AbstractVector{Float64},alpha::AbstractVector{NTuple{2,Int64}}, coefs::AbstractVector{Float64})
    for i in 1:length(alpha) v[i] = T_alpha_i(alpha[i],x) end
    return sum(v .* coefs)
end
                    
function T_alpha!(v::AbstractVector{Float64},x::AbstractVector{Float64},alpha::AbstractVector{NTuple{5,Int64}}, coefs::AbstractVector{Float64})
    for i in 1:length(alpha) v[i] = T_alpha_i(alpha[i],x) end
    return sum(v .* coefs)
end

    
# collect terms for tensor products
function collect_alpha(m,d)
    if d == 1
        alpha = 0:m
    elseif d == 2
        alpha = Iterators.product(0:m,0:m) 

    elseif d == 3
        alpha = Iterators.product(0:m,0:m,0:m)

    elseif d == 4
        alpha = Iterators.product(0:m,0:m,0:m,0:m)

    elseif d == 5
        alpha = Iterators.product(0:m,0:m,0:m,0:m,0:m)

    elseif d == 6
        alpha = Iterators.product(0:m,0:m,0:m,0:m,0:m,0:m)

    elseif d == 7
        alpha = Iterators.product(0:m,0:m,0:m,0:m,0:m,0:m,0:m)
    elseif d == 8
        alpha = Iterators.product(0:m,0:m,0:m,0:m,0:m,0:m,0:m,0:m)
    end

    alpha = reshape(collect(alpha),(m+1)^d)
    #alpha = broadcast(a -> broadcast(i -> a[i], 1:d), alpha)
    alpha = alpha[sum.(alpha) .<= m]
    return alpha
end
    

    
# collect z grid 
# creates and array of that stores
# the nodes for each dimension 
function collect_z(m,d)
    f = n -> -cos((2*n-1)*pi/(2*m))
    z = f.(1:m)
    if d == 1
        z_array = z
    elseif d == 2
        z_array = Iterators.product(z,z)  
    elseif d == 3
        z_array = Iterators.product(z,z,z) 
    elseif d == 4
        z_array = Iterators.product(z,z,z,z)
    elseif d == 5
        z_array = Iterators.product(z,z,z,z,z) 
    elseif d == 6
        z_array = Iterators.product(z,z,z,z,z,z)
    elseif d == 7
        z_array = Iterators.product(z,z,z,z,z,z,z)
    elseif d == 8
        z_array = Iterators.product(z,z,z,z,z,z,z) 
    end
    z_array = reshape(collect(z_array),m^d)
    z_array = broadcast(z -> broadcast(i -> z[i], 1:d), z_array)
    return z_array
end
    
    
    
# collect z grid 
# creates and array of that stores
# the nodes for each dimension 
function collect_nodes(nodes)
    d = size(nodes)[2]
    m = length(nodes[:,1])
    if d == 1
        z_array = z
    elseif d == 2
        z_array = Iterators.product(nodes[:,1],nodes[:,2])  
    elseif d == 3
        z_array = Iterators.product(nodes[:,1],nodes[:,2],nodes[:,3]) 
    elseif d == 4
        z_array = Iterators.product(nodes[:,1],nodes[:,2],nodes[:,3],nodes[:,4])
    elseif d == 5
        z_array = Iterators.product(nodes[:,1],nodes[:,2],nodes[:,3],nodes[:,4],nodes[:,5]) 
    elseif d == 6
        z_array = Iterators.product(nodes[:,1],nodes[:,2],nodes[:,3],nodes[:,4],nodes[:,5],nodes[:,6]) 
    elseif d == 7
        z_array = Iterators.product(nodes[:,1],nodes[:,2],nodes[:,3],nodes[:,4],nodes[:,5],nodes[:,6],nodes[:,7]) 
    elseif d == 8
        z_array = Iterators.product(nodes[:,1],nodes[:,2],nodes[:,3],nodes[:,4],nodes[:,5],nodes[:,6],nodes[:,7],nodes[:,8]) 
    end
    z_array = reshape(collect(z_array),m^d)
    z_array = broadcast(z -> broadcast(i -> z[i], 1:d), z_array)
    return z_array
end

function compute_coefs(d,m,values,alpha,T_alpha_i_grid)
    # get size of 
    #@assert length(values) == m^d
    #@assert all(size(values) .== m)
    # get constants for each term
    d_bar = broadcast(alpha_i -> sum(alpha_i .> 0), alpha)
    c = 2 .^ d_bar ./(m^d)
    # compute sum for each term
    z = collect_z(m,d) 
    # this is complicated. it initally broadcasts T_alpha_i over the grid z and takes the dot 
    # product with values array. It then broad casts this function over each set of valeus in alpha
    
    vT_sums = broadcast(alpha_i -> sum(broadcast(x -> T_alpha_i(alpha_i,x), z).*values), alpha)   
    
    return c.*vT_sums
end 
    
function compute_coefs1(d,m,values::AbstractVector{Float64},alpha::AbstractArray{NTuple{2,Int64}},T_alpha_grid::AbstractVector{AbstractVector{Float64}},
        vT_sums::AbstractVector{Float64})
    # get constants for each term
    d_bar = broadcast(alpha_i -> sum(alpha_i .> 0), alpha)
    c = 2 .^ d_bar ./(m^d)
    # this is complicated. it initally broadcasts T_alpha_i over the grid z and takes the dot 
    # product with values array. It then broad casts this function over each set of valeus in alpha 
    for i in 1:length(T_alpha_grid) vT_sums[i] =  sum(T_alpha_grid[i].*values) end 
    return c.*vT_sums
end
            
function compute_coefs1(d,m,values::AbstractVector{Float64},alpha::AbstractArray{NTuple{5,Int64}},T_alpha_grid::AbstractVector{AbstractVector{Float64}},
        vT_sums::AbstractVector{Float64})
    # get constants for each term
    d_bar = broadcast(alpha_i -> sum(alpha_i .> 0), alpha)
    c = 2 .^ d_bar ./(m^d)
    # this is complicated. it initally broadcasts T_alpha_i over the grid z and takes the dot 
    # product with values array. It then broad casts this function over each set of valeus in alpha 
    for i in 1:length(T_alpha_grid) vT_sums[i] =  sum(T_alpha_grid[i].*values) end 
    return c.*vT_sums
end

    
# x is a d by nx matrix 
function regression_matrix(x, polynomial)
    # transform from [a,b] to [-1,1]
    a = polynomial.a
    b = polynomial.b   
    z = mapslices(xi -> (xi.-a).*2 ./(b.-a) .- 1,x;dims = 2)
    f_x = xi -> broadcast(a -> utils.T_alpha_i(a,xi),polynomial.alpha)
    m = mapslices(f_x, z; dims=2)
    return  m
end 
    

end