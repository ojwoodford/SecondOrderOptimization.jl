!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)
using Test, LinearAlgebra, SparseArrays, StaticArrays, SecondOrderOptimization, BlockLinearSystems
using DampedLinearSolve: systemeltype, getA, getb, getx, getr, makehessiansystem
include("fixtures/load.jl"); loadfixtures("schur_problems.jl", "tp3.jl")
const SOO = SecondOrderOptimization

# A Float32 prototype (single-precision.md, SP-S3): every iterator on every conditioned system kind, both paths,
# converges to the Float64 run's cost within 1e-6 relative (rules.md 31); the iterations of both are logged. Nothing
# else in the loop is Float32 (the gradient, the step, λ, the costs), and a warm run with a workspace allocates
# nothing.

# A smooth objective whose Hessian has the pattern of a SchurTestProblem: f(x) = ½xᵀHx − bᵀx + (c/4)Σx⁴, so that
# ∇²f = H + 3c·Diagonal(x²), positive definite, with its minimum away from 0
struct PrQuartic
    p::SchurTestProblem
    c::Float64
    Hx::SchurTestProblem   # p with the curvature of the quartic term added to the diagonal, rewritten by each fdf!
end
PrQuartic(p, c) = PrQuartic(p, c, SchurTestProblem(copy(p.H), p.b, p.r_dims, p.e_dims, p.couplings, p.rr_pairs))
(q::PrQuartic)(x) = 0.5 * dot(x, q.p.H, x) - dot(q.p.b, x) + 0.25 * q.c * sum(v -> v^4, x)
function pr_grad!(g, q::PrQuartic, x)
    mul!(g, q.p.H, x)
    @inbounds for i in eachindex(g)
        g[i] += q.c * x[i]^3 - q.p.b[i]
    end
    @inbounds for i in eachindex(x)
        q.Hx.H[i, i] = q.p.H[i, i] + 3 * q.c * x[i]^2
    end
    return q(x)
end
struct PrFdf{K}
    q::PrQuartic
end
function (f::PrFdf{:dense})(g, H, x)
    cost = pr_grad!(g, f.q, x)
    H .= f.q.Hx.H
    return cost
end
function (f::PrFdf{:sparse})(g, H, x)
    cost = pr_grad!(g, f.q, x)
    rv, nz = rowvals(H), nonzeros(H)
    @inbounds for j in axes(H, 2), k in nzrange(H, j)
        nz[k] = f.q.Hx.H[rv[k], j]
    end
    return cost
end
# fill_schur! with the block ranges computed once, so that fdf! allocates nothing
struct PrSchurFdf
    q::PrQuartic
    rg::Vector{UnitRange{Int}}
end
function (f::PrSchurFdf)(g, H, x)
    cost = pr_grad!(g, f.q, x)
    p, rg = f.q.Hx, f.rg
    nrb = length(p.r_dims)
    for b in eachindex(rg)
        addblock!(H, b, b, p.H, rg[b], rg[b])
    end
    for (e, coupled) in enumerate(p.couplings), r in coupled
        addblock!(H, r, nrb + e, p.H, rg[r], rg[nrb + e])
    end
    return cost
end

pr_prototype(::Type{T}, kind, p) where T =
    kind === :dense ? zeros(T, size(p.H)) :
    kind === :sparse ? SparseMatrixCSC{T, Int}(sparse(p.H)) :
    fill_schur!(SchurBlockHessian{T, kind === :schurdense ? Matrix{T} : SparseBlockLinearSystem{T}}(schur_structure(p)), p)
pr_fdf(kind, q) = kind in (:schurdense, :schursparse) ? PrSchurFdf(q, block_ranges(dense_schur_matrix(q.p))) : PrFdf{kind}(q)
# :auto, except where it does not offer Float32 on this Julia version (CHOLMOD on SparseArrays 1.10)
pr_linsolver(kind) = kind === :schursparse && !(Float32 <: SparseArrays.CHOLMOD.VTypes) ? :ldl : :auto

const PR_ITERATORS = (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg)

@testset "precision.jl: the Hessian path with a Float32 prototype" begin
    p = in_layout(standard_problem())
    q = PrQuartic(p, 0.1)
    x0 = fill(0.5, length(p.b))
    for kind in (:dense, :sparse, :schurdense, :schursparse), it in PR_ITERATORS
        opts = OptimizationOptions(iterator = it, linsolver = pr_linsolver(kind))
        x64, r64 = optimize(copy(x0), HessianObjective(q, pr_fdf(kind, q); H = pr_prototype(Float64, kind, p)), opts)
        obj32 = HessianObjective(q, pr_fdf(kind, q); H = pr_prototype(Float32, kind, p))
        x32, r32 = optimize(copy(x0), obj32, opts)
        @info "Float32 Hessian" kind it iterations64 = r64.niterations iterations32 = r32.niterations cost64 = r64.bestcost cost32 = r32.bestcost
        @test isconverged(r64) && isconverged(r32)
        @test abs(r32.bestcost - r64.bestcost) <= 1e-6 * abs(r64.bestcost)
        @test eltype(x32) === Float64 && r32.bestcost isa Float64
        # The system is Float32; the gradient, the step and the damping are not
        ws = OptimizationWorkspace(copy(x0), obj32, opts)
        data, iteratedata = ws.data
        @test systemeltype(data.linsystem) === Float32
        @test eltype(getb(data.linsystem)) === Float64 && eltype(getx(data.linsystem)) === Float64
        hasproperty(iteratedata, :lambda) && @test iteratedata.lambda isa Float64
        # Zero allocation per iteration: a warm run with the workspace allocates nothing (the sparse
        # Schur system's CHOLMOD allocates inside SuiteSparse, rules.md 21)
        if !(kind === :schursparse && pr_linsolver(kind) === :auto)
            warm(x, obj, o, w) = @allocated optimize!(x, obj, o; workspace = w)
            warm(copy(x0), obj32, opts, ws)
            @test warm(copy(x0), obj32, opts, ws) == 0
        end
    end
    # A workspace built for one element type is not reused for another
    opts = OptimizationOptions()
    ws = OptimizationWorkspace(copy(x0), HessianObjective(q, pr_fdf(:dense, q); H = pr_prototype(Float32, :dense, p)), opts)
    @test !SOO.workspacevalid(ws, copy(x0), opts, pr_prototype(Float64, :dense, p))
end

# The Jacobian path: residuals r(x) = A x − y + (c/3)·x.^3 stacked under them (m = 2n), J = [A; c·Diagonal(x.^2)]
struct PrLsq
    A::Matrix{Float64}
    y::Vector{Float64}
    c::Float64
    rbuf::Vector{Float64}   # for the cost, so that f allocates nothing
end
PrLsq(A, y, c) = PrLsq(A, y, c, zeros(2 * size(A, 2)))
function pr_residual!(r, p::PrLsq, x)
    n = length(x)
    mul!(view(r, 1:n), p.A, x)
    @inbounds for i in 1:n
        r[i] -= p.y[i]
        r[n + i] = p.c / 3 * x[i]^3
    end
    return 0.5 * sum(abs2, r)
end
(p::PrLsq)(x) = pr_residual!(p.rbuf, p, x)
struct PrResJac
    p::PrLsq
end
function (f::PrResJac)(r, J, x)
    n = length(x)
    cost = pr_residual!(r, f.p, x)
    @inbounds for j in 1:n, i in 1:n
        J[i, j] = f.p.A[i, j]
        J[n + i, j] = i == j ? f.p.c * x[i]^2 : 0
    end
    return cost
end

@testset "precision.jl: the Jacobian path with a Float32 prototype" begin
    n = 8
    A = [sin(0.9 * i + 1.1 * j + 0.2 * i * j) + (i == j ? 3.0 : 0.0) for i in 1:n, j in 1:n]
    lsq = PrLsq(A, [cos(0.7 * i) for i in 1:n], 0.5)
    x0 = fill(0.3, n)
    for s in (:auto, :qr, :cholesky, :ldl), it in PR_ITERATORS
        opts = OptimizationOptions(iterator = it, linsolver = s)
        _, r64 = optimize(copy(x0), JacobianObjective(lsq, PrResJac(lsq), zeros(2n, n)), opts)
        obj32 = JacobianObjective(lsq, PrResJac(lsq), zeros(Float32, 2n, n))
        x32, r32 = optimize(copy(x0), obj32, opts)
        @info "Float32 Jacobian" s it iterations64 = r64.niterations iterations32 = r32.niterations cost64 = r64.bestcost cost32 = r32.bestcost
        @test isconverged(r64) && isconverged(r32)
        @test abs(r32.bestcost - r64.bestcost) <= 1e-6 * abs(r64.bestcost)
        ws = OptimizationWorkspace(copy(x0), obj32, opts)
        @test systemeltype(ws.data[1].linsystem) === Float32 && eltype(getr(getA(ws.data[1].linsystem))) === Float64
        warm(x, obj, o, w) = @allocated optimize!(x, obj, o; workspace = w)
        warm(copy(x0), obj32, opts, ws)
        @test warm(copy(x0), obj32, opts, ws) == 0
    end
    # The workspace key tells the element types apart
    opts = OptimizationOptions()
    ws = OptimizationWorkspace(copy(x0), JacobianObjective(lsq, PrResJac(lsq), zeros(Float32, 2n, n)), opts)
    @test !SOO.workspacevalid(ws, copy(x0), opts, SOO.jacobian_structure(zeros(2n, n)))
    @test SOO.workspacevalid(ws, copy(x0), opts, SOO.jacobian_structure(zeros(Float32, 2n, n)))
end

# The Jacobian path through QR elimination (SP-S3's rest; BlockLinearSystems SP-B1): a SchurBlockJacobian{Float32, R}
# prototype, TP3's nonlinear problem, dense and sparse reduced storage, every solver that offers Float32 (sparse: no
# SPQR; :cholesky where CHOLMOD offers Float32)
pr_qrsolvers(::Type{Matrix{Float32}}) = (:auto, :qr, :cholesky, :ldl)
pr_qrsolvers(::Type{SparseMatrixCSC{Float32, Int}}) = Float32 <: SparseArrays.CHOLMOD.VTypes ? (:cholesky, :ldl) : (:ldl,)
@testset "precision.jl: QR elimination with a SchurBlockJacobian{Float32} prototype" begin
    p = TP3Problem(tp3_structure())
    x0 = fill(0.1, p.lay.n)
    for R in (Matrix{Float32}, SparseMatrixCSC{Float32, Int}), s in pr_qrsolvers(R), it in PR_ITERATORS
        R64 = R <: Matrix ? Matrix{Float64} : SparseMatrixCSC{Float64, Int}
        opts = OptimizationOptions(iterator = it, linsolver = s)
        _, r64 = optimize(copy(x0), JacobianObjective(TP3Cost(p), TP3Tables(p), SchurBlockJacobian{Float64, R64}(p.jst)), opts)
        obj32 = JacobianObjective(TP3Cost(p), TP3Tables(p), SchurBlockJacobian{Float32, R}(p.jst))
        x32, r32 = optimize(copy(x0), obj32, opts)
        @info "Float32 QR elimination" R s it iterations64 = r64.niterations iterations32 = r32.niterations cost64 = r64.bestcost cost32 = r32.bestcost
        @test isconverged(r64) && isconverged(r32)
        @test abs(r32.bestcost - r64.bestcost) <= 1e-6 * abs(r64.bestcost)
        @test eltype(x32) === Float64 && r32.bestcost isa Float64
        ws = OptimizationWorkspace(copy(x0), obj32, opts)
        @test systemeltype(ws.data[1].linsystem) === Float32 && eltype(getb(ws.data[1].linsystem)) === Float64
        # Zero allocation per iteration on the dense reduced storage (the sparse one's CHOLMOD and LDLFactorizations
        # allocate inside the library, as in Float64)
        if R <: Matrix
            warm(x, obj, o, w) = @allocated optimize!(x, obj, o; workspace = w)
            warm(copy(x0), obj32, opts, ws)
            @test warm(copy(x0), obj32, opts, ws) == 0
        end
    end
end

# A problem whose variables differ in scale by 1e8 (SP-S3): the least-squares problem of the Jacobian testset above
# with its columns scaled from 1e-4 to 1e4, and the quartic with its variables scaled likewise. In Float64, Levenberg
# and dogleg take the same number of iterations to within 5% (at least 1) as before conditioning (the counts pinned are
# those of SecondOrderOptimization a42a5ae with DampedLinearSolve 3b16369 and BlockLinearSystems 0e446ca, measured by a
# script on macOS aarch64). The count is sensitive to rounding on this ill-conditioned problem: on Linux x64, dogleg with
# :auto takes 54 where macOS takes 52. In Float32 they converge.
function pr_scaled_lsq()
    n = 8
    sc = 10.0 .^ range(-4, 4; length = n)
    A = [sin(0.9 * i + 1.1 * j + 0.2 * i * j) + (i == j ? 3.0 : 0.0) for i in 1:n, j in 1:n] * Diagonal(sc)
    y = [cos(0.7 * i) for i in 1:n]
    res!(r, x) = (mul!(view(r, 1:n), A, x); r[1:n] .-= y; r[n+1:2n] .= 0.5 / 3 .* (x ./ sc) .^ 3; r)
    f(x) = 0.5 * sum(abs2, res!(zeros(2n), x))
    rj!(r, J, x) = (res!(r, x); J[1:n, :] .= A; J[n+1:2n, :] .= Diagonal(0.5 .* (x ./ sc) .^ 2 ./ sc); 0.5 * sum(abs2, r))
    return f, rj!, fill(0.3, n) ./ sc
end
const PR_SCALED_BEFORE = Dict(:levenbergmarquardt => 107, :dogleg => 52)
@testset "precision.jl: a problem whose variables differ in scale by 1e8" begin
    f, rj!, x0 = pr_scaled_lsq()
    n = length(x0)
    for it in (:levenbergmarquardt, :dogleg), s in (:auto, :cholesky)
        opts = OptimizationOptions(iterator = it, linsolver = s, maxiters = 300)
        _, r64 = optimize(copy(x0), JacobianObjective(f, rj!, zeros(2n, n)), opts)
        _, r32 = optimize(copy(x0), JacobianObjective(f, rj!, zeros(Float32, 2n, n)), opts)
        @info "1e8-scaled least squares" it s before = PR_SCALED_BEFORE[it] iterations64 = r64.niterations iterations32 = r32.niterations cost64 = r64.bestcost cost32 = r32.bestcost
        @test isconverged(r64) && abs(r64.niterations - PR_SCALED_BEFORE[it]) <= max(1, round(Int, 0.05 * PR_SCALED_BEFORE[it]))
        @test isconverged(r32)
    end
end

@testset "precision.jl: the constants that depend on the element type" begin
    for T in (Float64, Float32)
        ls = makehessiansystem(zeros(T, 2, 2), :auto)
        @test SOO.dampingcap(ls) == Float64(eps(T))^(3 / 4)
        @test SOO.lambdafloor(SOO.MarquardtData(0.0, zeros(2)), ls) == min(Float64(eps(T)), Float64(eps(T))^(3 / 4))
    end
    @test SOO.scalingfloor(Float64) == 1e-8
    @test SOO.scalingfloor(Float32) ≈ Float64(eps(Float32))^(1 / 2)
end
