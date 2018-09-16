"""
    PBVISolver <: Solver

POMDP solver type using the point-based value iteration algorithm.
"""
mutable struct PBVISolver <: Solver
    beliefpoints::Vector{Vector{Float64}}
    max_iterations::Int64
    tolerance::Float64
end

"""
    PBVISolver(; max_iterations, tolerance)

Initialize a point-based value iteration solver with the `max_iterations` limit and desired `tolerance`.
"""
function PBVISolver(;beliefpoints::Vector{Vector{Float64}}=[[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]], max_iterations::Int64=10, tolerance::Float64=1e-3)
    return PBVISolver(beliefpoints, max_iterations, tolerance)
end

"""
    AlphaVec

Alpha vector type of paired vector and action.
"""
struct AlphaVec
    alpha::Vector{Float64} # alpha vector
    action::Any # action associated wtih alpha vector
end

"""
    AlphaVec(vector, action_index)

Create alpha vector from `vector` and `action_index`.
"""
AlphaVec() = AlphaVec([0.0], 0)

# define alpha vector equality
==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))

"""
    create_policy(prune_solver, pomdp)

Create AlphaVectorPolicy for `prune_solver` using immediate rewards from `pomdp`.
"""
function create_policy(solver::PBVISolver, pomdp::Union{POMDP,RPOMDP})
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    alphas = [[reward(pomdp,S[i],A[j]) for i in 1:ns] for j in 1:na]
    AlphaVectorPolicy(pomdp, alphas)
end

function create_policy(solver::PBVISolver, pomdp::Union{IPOMDP,RIPOMDP})
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    b = fill(1 / ns, ns)
    alphas = [[reward(pomdp,b,A[j]) for i in 1:ns] for j in 1:na]
    AlphaVectorPolicy(pomdp, alphas)
end

# Eq. (5) Osogami 2015
# from p[z,a,t] to p[t,z,s]
function minutil(prob::Union{RPOMDP,RIPOMDP}, b::Vector{Float64}, a, alphavecs::Vector{Vector{Float64}})
    nz = n_observations(prob)
    ns = n_states(prob)
    nα = length(alphavecs)
    aind = RPOMDPModels.action_index(prob, a)
    plower, pupper = dynamics(prob)
    m = Model(solver = ClpSolver())
    @variable(m, u[1:nz])
    @variable(m, p[1:ns, 1:nz, 1:ns])
    @objective(m, Min, sum(u))
    for zind = 1:nz, αind = 1:nα
        @constraint(m, u[zind] >= sum(b[sind] * p[:,zind,sind]' * alphavecs[αind] for sind = 1:ns))
    end
    @constraint(m, p .<= pupper[:,:,:,aind])
    @constraint(m, p .>= plower[:,:,:,aind])
    for sind = 1:ns
        @constraint(m, sum(p[:,:,sind]) == 1)
    end
    JuMP.solve(m)
    getvalue(u), getvalue(p)
end

function findαz(zind::Int, u::Vector{Float64}, b::Vector{Float64}, p::Array{Float64}, alphavecs::Vector{Vector{Float64}})
    TOL = 1e-9
    αz = nothing
    ns = size(p, 1)
    for α in alphavecs
        s = sum(b[sind] * p[:,zind,sind]' * α for sind = 1:ns)
        if u[zind] ≈ sum(b[sind] * p[:,zind,sind]' * α for sind = 1:ns) atol = TOL
            αz = α
            break
        end
    end
    αz
end

function findαstar(rpomdp::Union{POMDP,RPOMDP}, b, s, a, pstar::Array{Float64}, alphaveczstar::Array{Array{Float64}})
    γ = rpomdp.discount
    ns = n_states(rpomdp)
    nz = n_observations(rpomdp)
    sind = state_index(rpomdp, s)
    αstars = reward(rpomdp, s, a)
    for tind in 1:ns, zind in 1:nz
        αstars += γ * pstar[tind, zind, sind] * alphaveczstar[zind][tind]
    end
    αstars
end

function findαstar(rpomdp::Union{IPOMDP,RIPOMDP}, b, s, a, pstar::Array{Float64}, alphaveczstar::Array{Array{Float64}})
    γ = rpomdp.discount
    ns = n_states(rpomdp)
    nz = n_observations(rpomdp)
    sind = state_index(rpomdp, s)
    αstars = rewardalpha(rpomdp, b, a)[sind]
    for tind in 1:ns, zind in 1:nz
        αstars += γ * pstar[tind, zind, sind] * alphaveczstar[zind][tind]
    end
    αstars
end

"""
    robustdpupdate(αset, beliefset, rpomdp)

Robust point-based dynamic programming backup value of `αset` for `beliefset` in `rpomdp`.
"""
function robustdpupdate(Vold::Set{AlphaVec}, beliefset::Vector{Vector{Float64}}, rp::Union{RPOMDP,RIPOMDP})
    alphaset = Set([avec.alpha for avec in Vold])
    Vnew = Set{AlphaVec}()
    # bcount = 0
    for b in beliefset
        Vbset = Set{AlphaVec}()
        for a in ordered_actions(rp)
            u, pstar = minutil(rp, b, a, collect(alphaset))
            αz = Array{Array{Float64}}(n_observations(rp))
            αstar = Vector{Float64}(n_states(rp))
            for (zind, z) in enumerate(ordered_observations(rp))
                αz[zind] = findαz(zind, u, b, pstar, collect(alphaset))
            end
            for (sind, s) in enumerate(ordered_states(rp))
                αstar[sind] = findαstar(rp, b, s, a, pstar, αz)
            end
            push!(Vbset, AlphaVec(αstar,a))
        end
        αmax = nothing
        vmax = -Inf
        for α in Vbset
            if b' * α.alpha > vmax
                vmax = b' * α.alpha
                αmax = α
            end
        end
        # bcount += 1
        # @show bcount
        # @show αmax
        Vnew = push!(Vnew, αmax)
    end
    Vnew
end

"""
    dpupdate(αset, beliefset, pomdp)

Point-based dynamic programming backup value of `αset` for `beliefset` in `pomdp`.
"""
function dpupdate(Vold::Set{AlphaVec}, beliefset::Vector{Vector{Float64}}, prob::Union{POMDP,IPOMDP})
    alphaset = Set([avec.alpha for avec in Vold])
    Vnew = Set{AlphaVec}()
    p = dynamics(prob)
    ns = n_states(prob)
    # bcount = 0
    for b in beliefset
        Vbset = Set{AlphaVec}()
        for (aind,a) in enumerate(ordered_actions(prob))
            αz = Array{Array{Float64}}(n_observations(prob))
            αstar = Vector{Float64}(n_states(prob))
            for (zind, z) in enumerate(ordered_observations(prob))
                smax = -Inf
                # TO DO: doublecheck definitoin of αnew
                for αnew in collect(alphaset)
                    s = sum(b[sind] * p[:,zind,sind,aind]' * αnew for sind = 1:ns)
                    if s > smax
                        smax = s
                        αz[zind] = αnew
                    end
                end
            end
            for (sind, s) in enumerate(ordered_states(prob))
                αstar[sind] = findαstar(prob, b, s, a, p[:,:,:,aind], αz)
            end
            push!(Vbset, AlphaVec(αstar,a))
        end
        αmax = nothing
        vmax = -Inf
        for α in Vbset
            if b' * α.alpha > vmax
                vmax = b' * α.alpha
                αmax = α
            end
        end
        # bcount += 1
        # @show bcount
        # @show αmax
        Vnew = push!(Vnew, αmax)
    end
    Vnew
end

"""
    diffvalue(Vnew, Vold, pomdp)

Maximum difference between new alpha vectors `Vnew` and old alpha vectors `Vold` in `pomdp`.
"""
function diffvalue(Vnew::Vector{AlphaVec}, Vold::Vector{AlphaVec}, pomdp::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP})
    ns = n_states(pomdp) # number of states in alpha vector
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    Anew = [avec.alpha for avec in Vnew]
    Aold = [avec.alpha for avec in Vold]
    dmax = -Inf # max difference
    for avecnew in Anew
        L = Model(solver = ClpSolver())
        @variable(L, x[1:ns])
        @variable(L, t)
        @objective(L, :Max, t)
        @constraint(L, x .>= 0)
        @constraint(L, x .<= 1)
        @constraint(L, sum(x) == 1)
        for avecold in Aold
            @constraint(L, (avecnew - avecold)' * x >= t)
        end
        sol = JuMP.solve(L)
        dmax = max(dmax, getobjectivevalue(L))
    end
    # rmin = minimum(reward(pomdp,s,a) for s in S, a in A) # minimum reward
    rmin = -1 # force second check since belief rewards have a more expensive min
    if rmin < 0 # if negative rewards, find max difference from old to new
        for avecold in Aold
            L = Model(solver = ClpSolver())
            @variable(L, x[1:ns])
            @variable(L, t)
            @objective(L, :Max, t)
            @constraint(L, x .>= 0)
            @constraint(L, x .<= 1)
            @constraint(L, sum(x) == 1)
            for avecnew in Anew
                @constraint(L, (avecold - avecnew)' * x >= t)
            end
            sol = JuMP.solve(L)
            dmax = max(dmax, getobjectivevalue(L))
        end
    end
    dmax
end

"""
    solve(solver::PBVISolver, rpomdp)

AlphaVectorPolicy for `pomdp` caluclated by the incremental pruning algorithm.
"""
function solve(solver::PBVISolver, prob::Union{RPOMDP,RIPOMDP})
    # println("Solver started...")
    ϵ = solver.tolerance
    replimit = solver.max_iterations
    policy = create_policy(solver, prob)
    avecs = [AlphaVec(policy.alphas[i], policy.action_map[i]) for i in 1:length(policy.action_map)]
    Vold = Set([AlphaVec(zeros(n_states(prob)), ordered_actions(prob)[1])])
    Vnew = Set{AlphaVec}()
    del = Inf
    reps = 0
    while del > ϵ && reps < replimit
        reps += 1
        Vnew = robustdpupdate(Vold, solver.beliefpoints, prob)
        del = diffvalue(collect(Vnew), collect(Vold), prob)
        Vold = Vnew
    end
    alphas_new = [v.alpha for v in Vnew]
    actions_new = [v.action for v in Vnew]
    policy = AlphaVectorPolicy(prob, alphas_new, actions_new)
    return policy
end

function solve(solver::PBVISolver, prob::Union{POMDP,IPOMDP})
    # println("Solver started...")
    ϵ = solver.tolerance
    replimit = solver.max_iterations
    policy = create_policy(solver, prob)
    avecs = [AlphaVec(policy.alphas[i], policy.action_map[i]) for i in 1:length(policy.action_map)]
    Vold = Set([AlphaVec(zeros(n_states(prob)), ordered_actions(prob)[1])])
    Vnew = Set{AlphaVec}()
    del = Inf
    reps = 0
    while del > ϵ && reps < replimit
        reps += 1
        Vnew = dpupdate(Vold, solver.beliefpoints, prob)
        del = diffvalue(collect(Vnew), collect(Vold), prob)
        Vold = Vnew
    end
    alphas_new = [v.alpha for v in Vnew]
    actions_new = [v.action for v in Vnew]
    policy = AlphaVectorPolicy(prob, alphas_new, actions_new)
    return policy
end

policyvalue(policy::AlphaVectorPolicy, b::Vector{Float64}) = maximum(dot(policy.alphas[i],b) for i in 1:length(policy.alphas))
