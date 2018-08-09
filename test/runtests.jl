using Base.Test
using RPOMDPModels
using RobustValueIteration
const PBVI = RobustValueIteration
TOL = 1e-6

@testset "Robust point-based value iteration" begin
    bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
    alphavecs = [[2.0, 4.0], [1.0, 5.0], [3.0, 3.0], [4.0, 2.0], [4.0, 3.0], [5.0, 1.0]]
    pupper = 0.8 + (rand(2,3,2) - 0.5)/10
    plower = 0.1 + (rand(2,3,2) - 0.5)/10
    pomdp = Baby3RPOMDP()
    b = bset[2]
    u, p = PBVI.minutil(b, pomdp, alphavecs)
    @show u
    @show p
    @test sum(u) ≈ 3.4 atol = TOL
    @test all(isapprox.(sum(p, (1,2)), 1.0, atol = TOL)) # p contains distributions

end # testset
