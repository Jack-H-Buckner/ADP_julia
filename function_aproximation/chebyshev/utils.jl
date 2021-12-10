"""
utils

defines some basic machinery used in both the tensor product and OLS_chebyshev
files. 

T(x,n) - degree n chebyshev polynomial 
T_alpha_i(alpha_i,x) - product of alpha_ith degree polynomial evalueated at x[i]
T_alpha(x,alpha, coefs)- Tensor product chebushex polynomial evaluated at x

"""
module utils


# the nth degree chebyshev polynomial
# evaluated at x
function T(x,n)
    return cos(n*acos(x))
end 

    
# the product of chebyshev polynomials
function T_alpha_i(alpha_i,x)
    return prod(T.(x,alpha_i)) 
end
    
# the product of chebyshev polynomials
# summed for each value of alpha
function T_alpha(x,alpha, coefs)
    T_i = broadcast(a -> T_alpha_i(a,x), alpha)
    return sum(T_i .* coefs)
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
    alpha = collect(alpha)[sum.(alpha) .<= m]
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
    return collect(z_array)
end

function compute_coefs(d,m,values,alpha)
    # get size of 
    @assert length(values) == m^d
    @assert all(size(values) .== m)
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

end # module 