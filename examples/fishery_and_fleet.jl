"""
defines function to simulate from a simple fishery model and adds a new state variable K_t
the defines the amount if investment in harvest equipment
"""
module fishery_and_fleet

using Distributions 

function T(s,a,pars)
    x, K = s[1], s[2]
    f, I = a[1], a[2]
    m, a, b, p, c, Delta, d = pars
    x = x*exp(-m-f) + a*x*exp(rand(d,1)[1])/(1+b*x)
    K = Delta*K + I
    return [x, K]
end

function profit(s,a,pars)
    x, K = s[1], s[2]
    f, I = a[1], a[2]
    m, a, b, p, c, d = pars
    return p*x*(1-exp(-f/(m+f))) - c/K*f
end 

sigma = 0.5
d = Distributions.Normal(0,sigma)
pars = (0.15, 2.0, 0.1, 10, 1000.0, 0.9, d)
delta = 0.95


end 