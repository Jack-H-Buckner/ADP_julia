"""
Defines a set of example models to test for consistency between 
"""
module Examples

using Distributions
#####################################
### Partially observable fishery ####
#####################################
"""
single period rewards
"""
function profit(f, catch_ ,pars)
    return pars[4]*catch_ - pars[5]*f
end 



"""
populaitn dynamics 

pars[1] - Float64 m natrual mortality rate
pars[2] - Float64 a recruitment rate
pars[3] - Float64 b strength of density dependence
pars[4] - Float64 price of fish
pars[5] - Float64 costs of fishing
pars[6] - Distribution d_recruits probability of recruitment
pars[7] - Distribution d_catch probability of catch
pars[8] - Distribution g_yx probability of observation log_y_t given log_x_t
pars[9] - Float64 weight_catch statistical weight of fisheries dependent data 
"""
function partially_observed_fishery_T!(x::AbstractVector{Float64},f::AbstractVector{Float64},pars)
    nu = rand(pars[10],1)[1]
    x .= exp(x[1])*exp(-pars[1]-f[1]+nu)
    x .= x[1]+ pars[2]*x[1]*exp(rand(pars[6],1)[1])/(1+pars[3]*x[1])
    x .= log.(x)
end


"""
Fisheries profit funciton 

pars[1] - Float64 m natrual mortality rate
pars[2] - Float64 a recruitment rate
pars[3] - Float64 b strength of density dependence
pars[4] - Float64 price of fish
pars[5] - Float64 costs of fishing
pars[6] - Distribution d_recruits probability of recruitment
pars[7] - Distribution d_catch probability of catch
pars[8] - Distribution g_yx probability of observation log_y_t given log_x_t
pars[9] - Float64 weight_catch statistical weight of fisheries dependent data 
"""
function partially_observed_fishery_R!(x,f,pars)
    x = x[1]
    x = exp(x)
    m = pars[1]
    f = f[1]
    catch_ = x *(1-exp(-f/(m+f)))*exp(rand(pars[7],1)[1])  
    return  profit(f, catch_ ,pars)
end


"""
log likelihood of x given 
"""
function partially_observed_fishery_G(x,y,f,pars)
    log_Y_t, log_c_t = y[1], y[2]
    L_y = pdf.(pars[8], x[1] - log_Y_t)
    L_c = pdf.(pars[7], x[1] - (log_c_t - log(1-exp(-f[1]/(pars[1]+f[1])))))
    L = L_y*L_c^pars[9]
    return L
end


"""
log likelihood of x given 
"""
function partially_observed_fishery_G_sim(x,f,pars)
    m = pars[1]
    x = x[1]
    f = f[1]
    log_Y_t, log_c_t = x + rand(pars[8],1)[1],  x + rand(pars[7],1)[1]+log(1-exp(-f/(m+f)))
    return [log_Y_t, log_c_t]
end



## paramters
m = 0.3#pars[1] - Float64 m natrual mortality rate
a = 1.5#pars[2] - Float64 a recruitment rate
b = 0.5#pars[3] - Float64 b strength of density dependence
p = 10 #pars[4] - Float64 price of fish
c = 2.0 #pars[5] - Float64 costs of fishing
sigma = 0.75
sigma_mort = 0.01
sigma_catch = 0.05
simga_obs = 1.0
d_recruits = Distributions.Normal(0,sigma)#pars[6] - Distribution d_recruits probability of recruitment
d_mort = Distributions.Normal(0,sigma_mort)#pars[6] - Distribution d_recruits probability of recruitment
d_catch = Distributions.Normal(0,sigma_catch) #pars[7] - Distribution d_catch probability of catch
g_yx  = Distributions.Normal(0,simga_obs) #pars[8] - Distribution g_yx probability of observation log_y_t given log_x_t
weight_catch = 0.01 #pars[9] - Float64 weight_catch statistical weight of fisheries dependent data 
partiallyObservedFishery_pars = [m,a,b,p,c,d_recruits,d_catch,g_yx,weight_catch,d_mort]



##############################################################
### Partially observable fishery unscented Kalman filters ####
##############################################################
"""
    unknownGrowthRate_T!(x,f,pars)

deterministic component of fishery model 
- x log popualtion size, growth rate
- f fishing mortality rate
- pars paramters including [1] ntrual mortality, [2] density dependence 
"""
function unknown_growth_rate_T!(x::AbstractVector{Float64},f::AbstractVector{Float64},pars)
    x[2] = pars[1] *x[2] + pars[2]
    k = pars[3]
    x[1] = x[1] + x[2] - f[1] - log(1+exp(x[2] +x[1]-f[1])/k)
    return x
end


"""
    unknownGrowthRate_T!(x,f,pars)

deterministic component of fishery model 
- x log popualtion size, growth rate
- f fishing mortality rate
- pars paramters including [1] ntrual mortality, [2] density dependence 
"""
function unknown_growth_rate_T(x::AbstractVector{Float64},f::AbstractVector{Float64},pars)
    r = pars[1] *x[2] + pars[2]
    k = pars[3]
    x1 = x[1] + r - f[1] - log(1+exp(r + x[1]-f[1])/k)
    return [x1,r]
end

Sigma_N = [0.01 0.0;
     0.0 0.02] # process noise

H = [1.0 0.0] # measurement model 
Sigma_O(sigma_t) = sigma_t # observation noise 

fmax(pars) = log(pars[2])

end # module 