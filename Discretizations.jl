"""
Defines structs and methods to convert and continuous state POMDP into a
discrete MDP. The general idea is to break the beleif state up into a finite
number a states and to calcualte the probability of transitioning between each
of these states conditional on the action taken. 

We can also calcualte the expected rewards for each beleif state aciton pair to
convert the POMDP into an MDP which can then be solved with standard dynamic 
programming techniques like VFI and Policy iteration. 

structs:

Discretization - Provides the data required to map from the beleif space (R^d x {1,2, ... ,n}) 
                 to the discretizd space ({1,2, ..., m}).
DiscreteMDP - Stores the data required to define a MDP (an array of state transitions and a matrix of expected rewards)

function:
Discretize - computes DiscreteMDP object from a POMDP and Discretization object 
initDiscretization - initalized discretization object from a minimal set of inputs
"""
module Discretizations

end # module