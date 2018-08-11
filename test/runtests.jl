using Base.Test
using RPOMDPModels
using RobustValueIteration
const PBVI = RobustValueIteration
TOL = 1e-6

@testset "Robust point-based value iteration" begin
    # minutil
    # Eq. (5) Osogami 2015
    srand(429)
    bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
    alphavecs = [[2.0, 4.0], [1.0, 5.0], [3.0, 3.0], [4.0, 2.0], [4.0, 3.0], [5.0, 1.0]]
    rpomdp = Baby3RPOMDP()
    b = bset[2]
    a = true
    u, pstar = PBVI.minutil(rpomdp, b, a, alphavecs)
    @test sum(u) ≈ 5 atol = TOL
    @test all(isapprox.(sum(pstar, (1,2)), 1.0, atol = TOL)) # p contains distributions

    # find αz that optimizes Eq. (5)
    # This is a way of making a non-linear program into two linear programs?
    nz = n_observations(rpomdp)
    αz = Array{Array{Float64}}(nz)
    for zind = 1:nz
        αz[zind] = PBVI.findαz(zind, u, b, pstar, alphavecs)
    end
    @test αz[1] == [1.0, 5.0]

    # find α*
    s = true
    a = false
    αstar = PBVI.findαstar(rpomdp, s, a, pstar, αz)
    @test αstar ≈ -5.5 atol = TOL

    alphaset = Set(alphavecs)
    αset = PBVI.robustdpval(alphaset, bset, rpomdp)
    @test length(αset) == 5
    @test pop!(αset)[1] ≈ -5.815 atol = TOL
end # testset


# using Base.Test
# using RPOMDPModels
# using RobustValueIteration
# const PBVI = RobustValueIteration
# TOL = 1e-6
#
# srand(429)
# bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
# alphavecs = [[2.0, 4.0], [1.0, 5.0], [3.0, 3.0], [4.0, 2.0], [4.0, 3.0], [5.0, 1.0]]
# rpomdp = Baby3RPOMDP()
#
# b = bset[2]
# a = true
# u, pstar = PBVI.minutil(rpomdp, b, a, alphavecs)
# nz = n_observations(rpomdp)
# αz = Array{Array{Float64}}(nz)
# for zind = 1:nz
#     αz[zind] = PBVI.findαz(zind, u, b, pstar, alphavecs)
# end
# s = true
# a = false
# αstar = PBVI.findαstar(rpomdp, s, a, pstar, αz)
#
# alphaset = Set(alphavecs)
# aset = PBVI.robustdpval(alphaset, bset, rpomdp)
