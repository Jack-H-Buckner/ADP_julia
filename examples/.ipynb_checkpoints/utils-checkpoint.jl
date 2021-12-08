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

end