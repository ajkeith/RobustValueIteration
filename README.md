# RobustValueIteration
Robust point-based value iteration algorithm for partially observable Markov decision processes (POMDPs), including standard reward and belief-reward versions. This application is built as solver for robust POMDPs. 

## Installation
This application is built for Julia 0.6. If not already installed, the application can be cloned using

```julia
Pkg.clone("https://github.com/ajkeith/RobustValueIteration")
```

To solve robust POMDP models, `RPOMDPs`, `RPOMDPToolbox`, `RPOMDPModels`, and `SimpleProbabilitySets` also need to be cloned

```julia
Pkg.clone("https://github.com/ajkeith/RPOMDPs.jl")
Pkg.clone("https://github.com/ajkeith/RPOMDPToolbox.jl")
Pkg.clone("https://github.com/ajkeith/RPOMDPModels.jl")
Pkg.clone("https://github.com/ajkeith/SimpleProbabilitySets.jl")
```

## Usage

This solver can be with the robust POMDP applications. See [RPOMDPs.jl](https://github.com/ajkeith/RPOMDPs.jl), [RPOMDPToolbox.jl](https://github.com/ajkeith/RPOMDPToolbox.jl), [RPOMDPModels.jl](https://github.com/ajkeith/RPOMDPModels.jl), and [SimpleProbabilitySets.jl](https://github.com/ajkeith/SimpleProbabilitySets.jl) for details on writing robust POMDP problems and associated tools. 

```julia
using RobustValueIteration
using RPOMDPModels, RPOMDPs, RPOMDPToolbox, SimpleProbabilitySets

rpomdp = RockRIPOMDP()
b = [psample(zeros(4), ones(4)) for i = 1:10]
solver = RPBVISolver(beliefpoints = b, max_iterations = 10)
policy = RobustValueIteration.solve(solver, rpomdp)
```

## References
The standard-reward robust point-based value iteration algorithm implements the pseudo code from Osogami (2015). The robust POMDP environment is a direct extension of [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl), [POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl), and [POMDPModelTools.jl](https://github.com/JuliaPOMDP/POMDPModelTools.jl) to the robust setting. 

If this code is useful to you, please star this package and consider citing the following papers.

Egorov, M., Sunberg, Z. N., Balaban, E., Wheeler, T. A., Gupta, J. K., & Kochenderfer, M. J. (2017). POMDPs.jl: A framework for sequential decision making under uncertainty. Journal of Machine Learning Research, 18(26), 1–5.

Osogami, T. (2015). Robust partially observable Markov decision process. In International Conference on Machine Learning (pp. 106–115).

