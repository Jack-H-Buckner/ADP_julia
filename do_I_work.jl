#import Pkg; Pkg.add("Distributions")
using Distributions

d = Distributions.Normal(0,1)
println(rand(d,100))
println("Do I work?")
