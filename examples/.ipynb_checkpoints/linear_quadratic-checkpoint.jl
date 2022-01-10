"""
simplest example 
AR1 process in one or two dimensions
"""
module linear_quadratic

using Distributions

function T!(x,c,pars)
    x = pars[1]*x .+ c .+ rand(pars[3],1)
    return x
end


function G(x,y,c,pars)
    return pdf(pars[4],y .- pars[2]*x)[1]
end 

function simulate_y(x,pars)
    return rand(pars[4],1) .+ pars[2]*x
end 

pars1d = (-0.9,1.0,Distributions.Normal(0,1.0),Distributions.Normal(0,2.0))

pars2d = ([0.5 -0.5; 0.5 0.5],[1.0 0; 0 1.0], 
           Distributions.MvNormal([0,0],[1.0 0.0; 0.0 1.0 ]),
           Distributions.MvNormal([0,0],[1.0 0.0; 0.0 1.0 ]))

pars_mixed = ([0.5 -0.5; 0.5 0.5],[1.0 0], 
           Distributions.MvNormal([0,0],[1.0 0.0; 0.0 1.0 ]),
           Distributions.Normal(0,1.0))

end 