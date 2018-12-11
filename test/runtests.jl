using Base.Test
using RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
const RPBVI = RobustValueIteration
TOL = 1e-6

@testset "Robust point-based value iteration" begin
    # value function max difference
    rp = BabyRPOMDP()
    alphavecs = [RPBVI.AlphaVec([2.0, 4.0], true),
                    RPBVI.AlphaVec([1.0, 5.0], :nothing),
                    RPBVI.AlphaVec([3.0, 3.0], :nothing),
                    RPBVI.AlphaVec([4.0, 2.0], :nothing),
                    RPBVI.AlphaVec([4.0, 3.0], :feed),
                    RPBVI.AlphaVec([5.0, 1.0], :nothing)]
    av2 = [RPBVI.AlphaVec([2.0, 4.0], true),
                    RPBVI.AlphaVec([1.1, 5.0], :nothing),
                    RPBVI.AlphaVec([3.0, 3.0], :nothing),
                    RPBVI.AlphaVec([4.0, 2.0], :nothing),
                    RPBVI.AlphaVec([4.0, 3.0], :feed),
                    RPBVI.AlphaVec([5.0, 0.9], :nothing)]
    @test RPBVI.diffvalue(alphavecs, av2, rp) ≈ 0.1 atol = TOL

    # minimum probability distribution (minutil)
    srand(429)
    bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
    alphas = [avec.alpha for avec in alphavecs]
    rip = BabyRIPOMDP()
    b = bset[2]
    a = :feed
    u, pstar = RPBVI.minutil(rip, b, a, alphas)
    @test sum(u) ≈ 4.8925 atol = TOL
    @test all(isapprox.(sum(pstar, (1,2)), 1.0, atol = TOL)) # p contains distributions

    # find αz that optimizes Eq. (5)
    nz = n_observations(rip)
    αz = Array{Array{Float64}}(nz)
    for zind = 1:nz
        αz[zind] = RPBVI.findαz(zind, u, b, pstar, alphas)
    end
    @test αz[1] == [1.0, 5.0]

    # find α*
    s = :hungry
    a = :nothing
    αstar = RPBVI.findαstar(rip, b, s, a, pstar, αz)
    @test αstar ≈ 3.40325 atol = TOL

    # robust point based dp backup
    srand(257349)
    αset = RPBVI.robustdpupdate(alphavecs, bset, rip)
    @test length(αset) == 6
    @test αset[1].alpha[1] ≈ 3.40325 atol = TOL

    # solver
    srand(429)
    bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
    p = BabyPOMDP()
    ip = BabyIPOMDP()
    rp = BabyRPOMDP()
    rip = BabyRIPOMDP()
    solver = RPBVISolver()
    solp = RPBVI.solve(solver, p)
    solip = RPBVI.solve(solver, ip)
    solrp = RPBVI.solve(solver, rp)
    solrip = RPBVI.solve(solver, rip)
    @test solp.action_map == [:nothing, :nothing, :feed, :feed, :feed, :feed]
    @test solip.action_map == [:feed, :feed, :feed, :feed, :feed, :nothing]
    @test solrp.action_map == [:nothing, :nothing, :feed, :feed, :feed, :feed]
    @test solrip.action_map == [:feed, :feed, :feed, :feed, :feed, :nothing]

    # beleif updates
    b = SparseCat([:hungry, :full], [0.4, 0.6])
    @test update(updater(solp),b,:nothing,:crying).b[1] ≈ 0.8363636363636364 atol = TOL
    @test update(updater(solrp),b,:nothing,:crying).b[1] ≈ 0.8497474114178433 atol = TOL

    # simulate
    sim = RolloutSimulator(max_steps = 1000)
    pbu = updater(solp)
    ipbu = updater(solip)
    rpbu = updater(solrp)
    ripbu = updater(solrip)
    @test simulate(sim, p, solp, pbu)[1] ≈ -20.29439154412 atol = 1
    @test simulate(sim, ip, solip, ipbu)[1] ≈ 10.0 atol = 1
    @test simulate(sim, rp, solrp, rpbu)[1] ≈ -48.60001793571622 atol = 1
    @test simulate(sim, rip, solrip, ripbu)[1] ≈ 4.908130152129587 atol = 1
end # testset

# Oracle values for ambiguity = 0 are from SARSOP
# Oracle values for ambiguity = 0.001, 0.1 from RobustInfoPOMDP/correctness_test.jl
@testset "SARSOP Comparison: RPOMDP" begin
    vtol = 0.1
    srand(8473272)
    p1 = TigerPOMDP(0.95)
    p1_001 = TigerRPOMDP(0.95, 0.001)
    p1_1 = TigerRPOMDP(0.95, 0.1)
    p2 = Baby2POMDP(-5.0, -10.0, 0.9)
    p2_001 = BabyRPOMDP(-5.0, -10.0, 0.9, 0.001)
    p2_1 = BabyRPOMDP(-5.0, -10.0, 0.9, 0.1)
    bs = [[b, 1-b] for b in 0.0:0.05:1.0]
    maxiter = 100
    solver = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)
    pol1 = RPBVI.solve(solver, p1)
    pol1_001 = RPBVI.solve(solver, p1_001)
    pol1_1 = RPBVI.solve(solver, p1_1)
    pol2 = RPBVI.solve(solver, p2)
    pol2_001 = RPBVI.solve(solver, p2_001)
    pol2_1 = RPBVI.solve(solver, p2_1)
    val1 = value(pol1, [0.5, 0.5])
    val1_001 = value(pol1_001, [0.5, 0.5])
    val1_1 = value(pol1_1, [0.5, 0.5])
    val2 = value(pol2, [0.0, 1.0])
    val2_001 = value(pol2_001, [0.0, 1.0])
    val2_1 = value(pol2_1, [0.0, 1.0])
    @test val1 ≈ 19.25 atol = vtol
    @test val1_001 ≈ 18.78 atol = vtol
    @test val1_1 ≈ -15.55 atol = vtol
    @test val2 ≈ -16.30 atol = vtol
    @test val2_001 ≈ -18.49 atol = vtol
    @test val2_1 ≈ -40.86 atol = vtol
end
