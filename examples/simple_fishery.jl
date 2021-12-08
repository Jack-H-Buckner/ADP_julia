"""
defines function to simulate from a simple fishery model
for an example of dynamic progaming.
"""
module simple_fishery
using Distributions 

function T(x,f,pars)
    m, a, b, p, c, d_recruits,d_catch = pars
    f = exp(rand(d_catch,1)[1])*f
    catch_ = x*(1-exp(-f/(m+f)))   
    return x*exp(-m-f) + a*x*exp(rand(d_recruits,1)[1])/(1+b*x),  catch_
end

function profit(f, catch_ ,pars)
    m, a, b, p, c, d_recruits,d_catch = pars
    return p*catch_ - c*f
end 

sigma = 0.5
sigma_catch = 0.1
d_recruits = Distributions.Normal(0,sigma)
d_catch = Distributions.Normal(0,sigma_catch)
pars = (0.15, 2.0, 0.1, 10, 0.1, d_recruits,d_catch)
delta = 0.95

end 