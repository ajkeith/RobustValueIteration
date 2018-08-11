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
    alphavecs = Set([PBVI.AlphaVec([2.0, 4.0], true),
                    PBVI.AlphaVec([1.0, 5.0], false),
                    PBVI.AlphaVec([3.0, 3.0], false),
                    PBVI.AlphaVec([4.0, 2.0], false),
                    PBVI.AlphaVec([4.0, 3.0], true),
                    PBVI.AlphaVec([5.0, 1.0], false)])
    alphas = [avec.alpha for avec in alphavecs]
    rpomdp = Baby3RPOMDP()
    b = bset[2]
    a = true
    u, pstar = PBVI.minutil(rpomdp, b, a, alphas)
    @test sum(u) ≈ 5 atol = TOL
    @test all(isapprox.(sum(pstar, (1,2)), 1.0, atol = TOL)) # p contains distributions

    # find αz that optimizes Eq. (5)
    # This is a way of making a non-linear program into two linear programs?
    nz = n_observations(rpomdp)
    αz = Array{Array{Float64}}(nz)
    for zind = 1:nz
        αz[zind] = PBVI.findαz(zind, u, b, pstar, alphas)
    end
    @test αz[1] == [1.0, 5.0]

    # find α*
    s = true
    a = false
    αstar = PBVI.findαstar(rpomdp, s, a, pstar, αz)
    @test αstar ≈ -5.5 atol = TOL

    αset = PBVI.robustdpupdate(alphavecs, bset, rpomdp)
    @test length(αset) == 5
    @test pop!(αset).alpha[1] ≈ -5.5 atol = TOL
end # testset


# using Base.Test
# using RPOMDPModels
# using RobustValueIteration
# const PBVI = RobustValueIteration
# TOL = 1e-6
#
# srand(429)
# bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
# alphavecs = Set([PBVI.AlphaVec([2.0, 4.0], true),
#                 PBVI.AlphaVec([1.0, 5.0], false),
#                 PBVI.AlphaVec([3.0, 3.0], false),
#                 PBVI.AlphaVec([4.0, 2.0], false),
#                 PBVI.AlphaVec([4.0, 3.0], true),
#                 PBVI.AlphaVec([5.0, 1.0], false)])
# alphas = [avec.alpha for avec in alphavecs]
# rpomdp = Baby3RPOMDP()
#
# b = bset[2]
# a = true
# u, pstar = PBVI.minutil(rpomdp, b, a, alphas)
# nz = n_observations(rpomdp)
# αz = Array{Array{Float64}}(nz)
# for zind = 1:nz
#     αz[zind] = PBVI.findαz(zind, u, b, pstar, alphas)
# end
# s = true
# a = false
# αstar = PBVI.findαstar(rpomdp, s, a, pstar, αz)
#
# aset = PBVI.robustdpupdate(alphavecs, bset, rpomdp)
#
# av1 = PBVI.AlphaVec([1.0, -1.0], 1)
# av2 = PBVI.AlphaVec([0.0, 1.0], 1)
# av3 = PBVI.AlphaVec([0.3, 0.2], 1)
# av4 = PBVI.AlphaVec([0.9, 0.9], 1)
# A = Set([av1,av2,av3,av4])
# alphaset = Set([avec.alpha for avec in A])
