using Base.Test
using RPOMDPModels
using RobustValueIteration
const PBVI = RobustValueIteration
TOL = 1e-6

@testset "Robust point-based value iteration" begin
    srand(429)
    bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
    alphavecs = Set([PBVI.AlphaVec([2.0, 4.0], true),
                    PBVI.AlphaVec([1.0, 5.0], false),
                    PBVI.AlphaVec([3.0, 3.0], false),
                    PBVI.AlphaVec([4.0, 2.0], false),
                    PBVI.AlphaVec([4.0, 3.0], true),
                    PBVI.AlphaVec([5.0, 1.0], false)])
    alphas = [avec.alpha for avec in alphavecs]
    rip = BabyRIPOMDP()
    b = bset[2]
    a = :feed
    u, pstar = PBVI.minutil(rip, b, a, alphas)
    @test sum(u) ≈ 4.8925 atol = TOL
    @test all(isapprox.(sum(pstar, (1,2)), 1.0, atol = TOL)) # p contains distributions

    # find αz that optimizes Eq. (5)
    # This is a way of making a non-linear program into two linear programs?
    nz = n_observations(rip)
    αz = Array{Array{Float64}}(nz)
    for zind = 1:nz
        αz[zind] = PBVI.findαz(zind, u, b, pstar, alphas)
    end
    @test αz[1] == [1.0, 5.0]

    # find α*
    s = :hungry
    a = :nothing
    αstar = PBVI.findαstar(rip, b, s, a, pstar, αz)
    @test αstar ≈ 3.40325 atol = TOL

    # robust point based dp backup
    srand(257349)
    αset = PBVI.robustdpupdate(alphavecs, bset, rip)
    @test length(αset) == 5
    @test pop!(αset).alpha[1] ≈ 4.60325 atol = TOL

    # solver
    srand(429)
    bset = [[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]]
    p = BabyPOMDP()
    ip = BabyIPOMDP()
    rp = BabyRPOMDP()
    rip = BabyRIPOMDP()
    solver = PBVISolver()
    solp = solve(solver, p)
    solip = solve(solver, ip)
    solrp = solve(solver, rp)
    solrip = solve(solver, rip)
    @test solp.action_map == [:feed, :nothing, :nothing]
    @test solip.action_map == [:feed, :feed, :nothing, :feed]
    @test solrp.action_map == [:feed, :nothing, :nothing]
    @test solrip.action_map == [:feed, :feed, :feed, :feed, :feed]
end # testset


using Base.Test
using RPOMDPModels
using RPOMDPToolbox
using RobustValueIteration
const PBVI = RobustValueIteration
TOL = 1e-6

p = BabyPOMDP()
ip = BabyIPOMDP()
rp = BabyRPOMDP()
rp2 = BabyRPOMDP(-5.0, -10.0, 0.9, 0.1)
rip = BabyRIPOMDP()


solver = PBVISolver(max_iterations = 10000, tolerance = 1e-3)
solp = RobustValueIteration.solve(solver, p)
solr = RobustValueIteration.solve(solver, rp)
solr2 = RobustValueIteration.solve(solver, rp2)
solr2
plot(solr2.alphas)

umin, pmin = PBVI.minutil(rp, [0.8, 0.2], :nothing, solr.alphas)


sim = RolloutSimulator(max_steps = 1000)
simulate(sim, rp, solr, DiscreteUpdater(p))

m = 1000
v = [simulate(sim,p,solp,DiscreteUpdater(p)) for i = 1:m]
# using Plots
plot(v)
mean(v)


uniform_belief(p)
b = SparseCat([:hungry, :full], [0.4, 0.6])
updater(solp)
updater(solr)
update(updater(solp),b,:nothing,:crying)
update(updater(solr),b,:nothing,:crying)
sim = RolloutSimulator()


using RPOMDPModels, RPOMDPToolbox
rng = MersenneTwister(120938)
p = BabyPOMDP()
generate_s(p, :full, :nothing, rng)
