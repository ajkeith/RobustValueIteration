"""
    RPBVISolver <: Solver

POMDP solver type using the point-based value iteration algorithm.
"""
mutable struct RPBVISolver <: Solver
    beliefpoints::Vector{Vector{Float64}}
    max_iterations::Int64
    tolerance::Float64
end

"""
    RPBVISolver(; max_iterations, tolerance)

Initialize a point-based value iteration solver with the `max_iterations` limit and desired `tolerance`.
"""
function RPBVISolver(;beliefpoints::Vector{Vector{Float64}}=[[0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6,0.4], [0.8,0.2], [1.0,0.0]], max_iterations::Int64=10, tolerance::Float64=1e-3)
    return RPBVISolver(beliefpoints, max_iterations, tolerance)
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
    create_policy(solver, pomdp)

Create AlphaVectorPolicy for `solver` using immediate rewards from `pomdp`.
"""
function create_policy(solver::RPBVISolver, pomdp::Union{POMDP,RPOMDP})
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    alphas = [[reward(pomdp,S[i],A[j]) for i in 1:ns] for j in 1:na]
    AlphaVectorPolicy(pomdp, alphas)
end

function create_policy(solver::RPBVISolver, pomdp::Union{IPOMDP,RIPOMDP})
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
    TOL = 1e-6
    αz = nothing
    ns = size(p, 1)
    for α in alphavecs
        s = sum(b[sind] * p[:,zind,sind]' * α for sind = 1:ns)
        # @show u[zind]
        # @show sum(b[sind] * p[:,zind,sind]' * α for sind = 1:ns)
        if isapprox(u[zind], sum(b[sind] * p[:,zind,sind]' * α for sind = 1:ns), atol = TOL)
            αz = α
            break
        end
    end
    # @show αz
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
function robustdpupdate(Vold::Vector{AlphaVec}, beliefset::Vector{Vector{Float64}}, rp::Union{RPOMDP,RIPOMDP})
    alphaset = [avec.alpha for avec in Vold]
    Vnew = Vector{AlphaVec}(size(beliefset, 1))
    @showprogress 1 "Updating..." for (bi, b) in enumerate(beliefset)
        Vbset = Set{AlphaVec}()
        for a in ordered_actions(rp)
            u, pstar = minutil(rp, b, a, alphaset)
            αz = Array{Array{Float64}}(n_observations(rp))
            αstar = Vector{Float64}(n_states(rp))
            for (zind, z) in enumerate(ordered_observations(rp))
                αz[zind] = findαz(zind, u, b, pstar, alphaset)
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
        Vnew[bi] = αmax
    end
    Vnew
end

"""
    dpupdate(αset, beliefset, pomdp)

Point-based dynamic programming backup value of `αset` for `beliefset` in `pomdp`.
"""
function dpupdate(Vold::Vector{AlphaVec}, beliefset::Vector{Vector{Float64}}, prob::Union{POMDP,IPOMDP})
    alphaset = Set([avec.alpha for avec in Vold])
    Vnew = Vector{AlphaVec}(size(beliefset, 1))
    p = dynamics(prob)
    ns = n_states(prob)
    @showprogress 1 "Updating..." for (bi, b) in enumerate(beliefset)
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
        Vnew[bi] = αmax
    end
    Vnew
end

"""
    diffvalue(Vnew, Vold, pomdp)

Maximum difference between new alpha vectors `Vnew` and old alpha vectors `Vold` in `pomdp`.
"""
function diffvalue(Vnew::Vector{AlphaVec}, Vold::Vector{AlphaVec}, pomdp::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP})
    nv = length(Vnew)
    ns = n_states(pomdp)
    dmax = -Inf
    for bi = 1:nv
        for si = 1:ns
            diff = abs(Vnew[bi].alpha[si] - Vold[bi].alpha[si])
            dmax = max(dmax, diff)
        end
    end
    dmax
end

"""
    solve(solver::RPBVISolver, rpomdp)

AlphaVectorPolicy for `pomdp` caluclated by the incremental pruning algorithm.
"""
function solve(solver::RPBVISolver, prob::Union{RPOMDP,RIPOMDP}; verbose::Bool=false)
    # println("Solver started...")
    ϵ = solver.tolerance
    iterlimit = solver.max_iterations
    policy = create_policy(solver, prob)
    avecs = [AlphaVec(policy.alphas[i], policy.action_map[i]) for i in 1:length(policy.action_map)]
    Vold = fill(AlphaVec(zeros(n_states(prob)), ordered_actions(prob)[1]), length(solver.beliefpoints))
    Vnew = Vector{AlphaVec}()
    del = Inf
    iter = 0
    while del > ϵ && iter < iterlimit
        iter += 1
        Vnew = robustdpupdate(Vold, solver.beliefpoints, prob)
        del = diffvalue(Vnew, Vold, prob)
        Vold = copy(Vnew)
        verbose && println("\riter: $iter, del: $del")
    end
    alphas_new = [v.alpha for v in Vnew]
    actions_new = [v.action for v in Vnew]
    policy = AlphaVectorPolicy(prob, alphas_new, actions_new)
    return policy
end

function solve(solver::RPBVISolver, prob::Union{POMDP,IPOMDP}; verbose::Bool=false)
    # println("Solver started...")
    ϵ = solver.tolerance
    iterlimit = solver.max_iterations
    policy = create_policy(solver, prob)
    avecs = [AlphaVec(policy.alphas[i], policy.action_map[i]) for i in 1:length(policy.action_map)]
    Vold = fill(AlphaVec(zeros(n_states(prob)), ordered_actions(prob)[1]), length(solver.beliefpoints))
    Vnew = Vector{AlphaVec}()
    del = Inf
    iter = 0
    while del > ϵ && iter < iterlimit
        iter += 1
        Vnew = dpupdate(Vold, solver.beliefpoints, prob)
        del = diffvalue(Vnew, Vold, prob)
        Vold = copy(Vnew)
        verbose && println("\riter: $iter, del: $del")
    end
    alphas_new = [v.alpha for v in Vnew]
    actions_new = [v.action for v in Vnew]
    policy = AlphaVectorPolicy(prob, alphas_new, actions_new)
    return policy
end

policyvalue(policy::AlphaVectorPolicy, b::Vector{Float64}) = maximum(dot(policy.alphas[i], b) for i in 1:length(policy.alphas))
