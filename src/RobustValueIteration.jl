module RobustValueIteration

using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using JuMP, Clp

import Base: ==, hash

export
    RPBVISolver,
    solve,
    policyvalue

include("rpbvi.jl")

end # module
