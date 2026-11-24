using Test, Random, LinearAlgebra, SparseArrays, StaticArrays, Static
import DampedLinearSolve
using SecondOrderOptimization, BlockLinearSystems, Polyester   # Polyester: BlockLinearSystems' threaded kernels
using SecondOrderOptimization: check_linear_system, getA, getx   # DampedLinearSolve's names, through SecondOrderOptimization
include("fixtures/load.jl"); loadfixtures("schur_problems.jl")

# The Schur-complement systems of a SchurBlockHessian (BlockLinearSystems' implementation of DampedLinearSolve's
# contract) under the iterators and the workspace, through BlockLinearSystems' public API and the contract only.
# Their solve-level tests, and what only BlockLinearSystems can see (which system a prototype builds, its reduced
# system), are BlockLinearSystems'.
# Every Schur solve is compared against a dense Cholesky solve of the full damped system.


@testset "BlockLinearSystemsExt: the type of H selects elimination; linsolver names the reduced solver (A2)" begin
    p = standard_problem()
    Hd, Hs = dense_schur_matrix(p), sparse_schur_matrix(p)
    mk(H, s) = SecondOrderOptimization.makehessiansystem(H, s)
    # The enum spellings of a solver reach the systems as the options hold them, as a StaticSymbol
    opt(s) = OptimizationOptions(linsolver = s).linsolver
    # Every spelling of a reduced-system solver builds a system, dense or sparse (which system the storage of R picks
    # is BlockLinearSystems' to test)
    for s in (:auto, :cholesky, :ldl, static(:auto), static(:ldl), opt(SecondOrderOptimization.auto_solver),
              opt(SecondOrderOptimization.cholesky_solver), opt(SecondOrderOptimization.ldl_solver))
        @test mk(Hd, s) isa SecondOrderOptimization.AbstractLinearSystem
        @test mk(Hs, s) isa SecondOrderOptimization.AbstractLinearSystem
    end
    # Solvers not implemented for the reduced system are rejected when it is built (not treated as :auto)
    sq(x) = sum(abs2, x)
    sq_fdf!(g, H, x) = sq(x)
    for H in (Hd, Hs)
        for s in (:qr, :pardiso, :cg, :minres, static(:qr), opt(SecondOrderOptimization.qr_solver))
            err = try mk(H, s); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("reduced system", sprint(showerror, err))
        end
        opts_qr = OptimizationOptions(linsolver=:qr)
        @test_throws ArgumentError optimize(zeros(length(p.b)), HessianObjective(sq, sq_fdf!; H), opts_qr)
        @test_throws ArgumentError OptimizationWorkspace(zeros(length(p.b)), HessianObjective(sq, sq_fdf!; H), opts_qr)
    end
    # ...and by solve! on a system built directly
    for kind in (:dense, :sparse)
        @test_throws ArgumentError SecondOrderOptimization.solve!(make_schur_ls(kind, p), :qr)
    end
    # The old symbols are unknown even with the extension loaded; other Hessians need no check
    for s in (:schur, :dense_schur, :sparse_schur)
        @test !SecondOrderOptimization.validatelinsolver(s)
        @test_throws ArgumentError OptimizationOptions(linsolver=s)
    end
    @test_throws MethodError OptimizationOptions(reduction=:eliminate)
    for H_other in (nothing, zeros(3, 3), sparse(1.0I, 3, 3))
        x, res = optimize(ones(3), HessianObjective(sq, (g, H, x) -> (g .= 2 .* x; H[diagind(H)] .= 2; sq(x)); H = H_other))
        @test isconverged(res) && norm(x) < 1e-8
    end

    # solve! accepts Symbols and static symbols (the enum spellings as the options hold them) without ambiguity
    n = length(p.b)
    x_ref = reference_solve(p, 0.3, ones(n))
    for kind in (:dense, :sparse), s in (:auto, :cholesky, :ldl, static(:auto), static(:ldl),
                                        opt(SecondOrderOptimization.auto_solver), opt(SecondOrderOptimization.cholesky_solver), opt(SecondOrderOptimization.ldl_solver))
        ls = make_schur_ls(kind, p; linsolver = s)
        SecondOrderOptimization.dampen!(ls, 0.3)
        @test SecondOrderOptimization.solve!(ls, s) == SecondOrderOptimization.Success
        @test norm(getx(ls) - x_ref) <= 1e-10 * norm(x_ref)
    end
end


# The same system in another element type: built from the same structure, its parts copied whole
function to_eltype(::Type{T}, H::SchurBlockHessian) where T
    HT = SchurBlockHessian{T, H.R isa Matrix ? Matrix{T} : SparseBlockLinearSystem{T}}(H.structure)
    H.R isa Matrix ? (HT.R .= H.R) : (HT.R.data .= H.R.data)
    HT.E_data .= H.E_data
    HT.C_data .= H.C_data
    return HT
end

# A SchurBlockHessian{Float32} gives a single-precision system (single-precision.md); another element type is rejected
# up front, before f or fdf! is called. (Until SP-S1 a Float32 prototype was rejected here too.)
@testset "A SchurBlockHessian must be Float64 or Float32; another element type is rejected up front" begin
    p = standard_problem()
    n = length(p.b)
    for kind in (:dense, :sparse)
        H = to_eltype(Float16, kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p))
        calls = Ref(0)
        obj = HessianObjective(x -> (calls[] += 1; 0.0), (g, Hs, x) -> (calls[] += 1; 0.0); H)
        for s in (:auto, :ldl)
            @test_throws ArgumentError optimize(zeros(n), obj, OptimizationOptions(; linsolver = s))
            @test_throws ArgumentError OptimizationWorkspace(zeros(n), obj, OptimizationOptions(; linsolver = s))
        end
        @test calls[] == 0
        err = try optimize(zeros(n), obj, OptimizationOptions()) catch e; e end
        @test occursin("SchurBlockHessian{Float32}", sprint(showerror, err))
        H32 = to_eltype(Float32, kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p))
        ws = OptimizationWorkspace(zeros(n), HessianObjective(x -> 0.0, (g, Hs, x) -> 0.0; H = H32), OptimizationOptions(; linsolver = :ldl))
        @test SecondOrderOptimization.systemeltype(ws.data[1].linsystem) === Float32   # DampedLinearSolve's, through SecondOrderOptimization
    end
end





# A gauge-deficient problem: every coupling constrains only the *difference* x_retained[1:3] - x_eliminated, so
# translating all retained and eliminated blocks together changes nothing and H is singular along that direction.
# Dimensions 4:6 of the retained blocks only have a tiny prior, which makes H ill conditioned.
function gauge_problem(seed=21; prior=1e-8)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 3, 3]
    couplings = [[1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [1, 4], [1, 2, 3, 4]]
    nr_, np_ = sum(r_dims), sum(e_dims)
    H = zeros(nr_ + np_, nr_ + np_)
    for (e, coupled) in enumerate(couplings), r in coupled
        w = 0.5 + rand(rng)
        ci = 6 * (r - 1) .+ (1:3)
        li = nr_ + 3 * (e - 1) .+ (1:3)
        H[ci, ci] .+= w * I(3); H[li, li] .+= w * I(3)
        H[ci, li] .-= w * I(3); H[li, ci] .-= w * I(3)
    end
    for r in 1:4
        ci = 6 * (r - 1) .+ (4:6)
        H[ci, ci] .+= prior * I(3)
    end
    return SchurTestProblem(H, randn(rng, nr_ + np_), r_dims, e_dims, couplings, Tuple{Int, Int}[])
end





# Add d to the diagonal of the full matrix a SchurBlockHessian represents: Diagonal(d)'s block of each layout block,
# whose scalar ranges `ranges` are (block_ranges(H), computed once where the call must not allocate)
function add_schur_diag!(H::SchurBlockHessian, d, ranges = block_ranges(H))
    D = Diagonal(d)
    for (b, r) in enumerate(ranges)
        addblock!(H, b, b, D, r, r)
    end
    return H
end

# A quadratic plus a quartic term, so the Hessian H + diag(3x^2) changes every iteration. `kind` is
# :dense or :sparse (a SchurBlockHessian with that storage of R) or :full (a plain dense Hessian).
function quartic_run(p, kind, linsolver, iterator)
    cost(x) = 0.5 * dot(x, p.H * x) - dot(p.b, x) + 0.25 * sum(xi -> xi^4, x)
    if kind === :full
        Hproto = zeros(length(p.b), length(p.b))
        fdf! = (g, H, x) -> (g .= p.H * x .- p.b .+ x .^ 3; H .= p.H; H[diagind(H)] .+= 3 .* x .^ 2; cost(x))
    else
        Href = kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p)
        Hproto = kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p)
        fdf! = (g, H, x) -> (g .= p.H * x .- p.b .+ x .^ 3; copy_content!(H, Href); add_schur_diag!(H, 3 .* x .^ 2); cost(x))
    end
    traj = CostTrajectory()
    x, res = optimize(fill(3.0, length(p.b)), HessianObjective(cost, fdf!; H = Hproto),
                      OptimizationOptions(; iterator, linsolver, gtol=1e-10); callback=storecostscallback(traj))
    return (; x, res, costs = copy(traj.costs), steps = deepcopy(traj.steps))
end

@testset "Full optimization with a SchurBlockHessian: :auto, :cholesky and :ldl (A2)" begin
    p = standard_problem()
    for kind in (:dense, :sparse), it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :newton, :dogleg)
        @testset "$kind, $it" begin
            ra = quartic_run(p, kind, :auto, it)
            rc = quartic_run(p, kind, :cholesky, it)
            rl = quartic_run(p, kind, :ldl, it)
            rf = quartic_run(p, :full, :cholesky, it)
            @test isconverged(ra.res) && isconverged(rl.res)
            @test ra.res.niterations > 3
            # :auto and :cholesky are the same solver for the reduced system: identical iterates
            @test ra.x == rc.x && ra.costs == rc.costs && ra.steps == rc.steps
            # the same optimum as :ldl and as the same problem with a plain dense Hessian
            @test norm(rl.x - ra.x) <= 1e-8 * norm(ra.x)
            @test norm(rf.x - ra.x) <= 1e-8 * norm(ra.x)
            @test ra.res.niterations == rf.res.niterations
        end
    end
end


# AS1: a workspace for a SchurBlockHessian. The quartic of `quartic_run`, evaluated without allocating
function as1_objective(p, Href)
    d = similar(p.b)
    ranges = block_ranges(Href)
    cost(x) = 0.5 * dot(x, p.H, x) - dot(p.b, x) + 0.25 * sum(xi -> xi^4, x)
    # Products, not `x .^ 3`: on Julia 1.10 with bounds checks on, a literal power in a broadcast can allocate a Ref
    function fdf!(g, H, x)
        mul!(g, p.H, x)
        g .+= x .* x .* x .- p.b
        copy_content!(H, Href)
        d .= 3 .* x .* x
        add_schur_diag!(H, d, ranges)
        return cost(x)
    end
    return cost, fdf!
end
as1_allocs(x, obj, o, w) = (x .= 3.0; @allocated optimize!(x, obj, o; workspace = w))
as1_valid_allocs(w, x, o, H) = @allocated SecondOrderOptimization.workspacevalid(w, x, o, H)

@testset "OptimizationWorkspace with a SchurBlockHessian (AS1)" begin
    p = standard_problem()
    n = length(p.b)
    opts = OptimizationOptions(; iterator = :levenbergmarquardt, gtol = 1e-10)
    for kind in (:dense, :sparse)
        schur(q) = kind === :dense ? dense_schur_matrix(q) : sparse_schur_matrix(q)
        H = schur(p)
        cost, fdf! = as1_objective(p, schur(p))
        obj = HessianObjective(cost, fdf!; H)
        x_ref, res_ref = optimize(fill(3.0, n), obj, opts)
        @test isconverged(res_ref)
        for typed in (true, false)
            @testset "$kind, $(typed ? "typed" : "untyped") workspace" begin
                ws = typed ? OptimizationWorkspace(fill(3.0, n), obj, opts) : OptimizationWorkspace()
                # The first solve with a workspace gives the result of a solve without one
                x1 = fill(3.0, n)
                res1 = optimize!(x1, obj, opts; workspace = ws)
                @test x1 == x_ref && res1.niterations == res_ref.niterations
                @test !isempty(ws)
                built = ws.data
                # A second solve with the same H reuses it. With a typed workspace, matching the key allocates
                # nothing, and the whole dense run allocates nothing, as for a dense Hessian (the sparse run's
                # CHOLMOD allocates, rules.md 21; an untyped workspace dispatches dynamically)
                x2 = fill(3.0, n)
                optimize!(x2, obj, opts; workspace = ws)
                @test ws.data === built && x2 == x_ref
                if typed
                    as1_valid_allocs(ws, x2, opts, H)
                    @test as1_valid_allocs(ws, x2, opts, H) == 0
                end
                if typed && kind === :dense
                    as1_allocs(x2, obj, opts, ws)
                    @test as1_allocs(x2, obj, opts, ws) == 0
                end
                # An H with another structure object rebuilds it, and still gives the right result
                H_other = schur(p)
                @test H_other.structure !== H.structure
                obj_other = HessianObjective(cost, fdf!; H = H_other)
                @test !SecondOrderOptimization.workspacevalid(ws, x2, opts, H_other)
                x3 = fill(3.0, n)
                optimize!(x3, obj_other, opts; workspace = ws)
                @test ws.data !== built
                @test x3 == x_ref
            end
        end
        # A workspace for a Schur H is not accepted for a dense H of the same size, nor the reverse
        Hd = zeros(n, n)
        ws_s = OptimizationWorkspace(fill(3.0, n), obj, opts)
        ws_d = OptimizationWorkspace(fill(3.0, n), HessianObjective(cost, (g, Hm, x) -> 0.0; H = Hd), opts)
        @test !SecondOrderOptimization.workspacevalid(ws_s, fill(3.0, n), opts, Hd)
        @test !SecondOrderOptimization.workspacevalid(ws_d, fill(3.0, n), opts, H)
        @test SecondOrderOptimization.workspacevalid(ws_s, fill(3.0, n), opts, H)
        # An x0 of the wrong size is a DimensionMismatch up front, before the objective is evaluated
        calls = Ref(0)
        obj_count = HessianObjective(cost, (g, Hm, x) -> (calls[] += 1; 0.0); H)
        @test_throws DimensionMismatch optimize!(fill(3.0, n + 1), obj_count, opts)
        @test_throws DimensionMismatch OptimizationWorkspace(fill(3.0, n - 1), obj_count, opts)
        @test calls[] == 0
        err = try optimize!(fill(3.0, n + 1), obj_count, opts) catch e; e end
        @test occursin("$(n + 1) degrees of freedom", sprint(showerror, err))
    end
end

@testset "updatescaling! guards a non-finite diagonal entry, as the other systems do (AS2)" begin
    p = standard_problem()
    n = length(p.b)
    for kind in (:dense, :sparse), bad in (NaN, Inf, -Inf), k in (1, nr(p) + 1)   # one retained, one eliminated entry
        @testset "$kind, $bad at $(k == 1 ? "retained" : "eliminated") entry" begin
            ls = make_schur_ls(kind, p)
            d = zeros(n)
            d[k] = bad
            add_schur_diag!(getA(ls), d)
            M = SecondOrderOptimization.conformance_matrix(getA(ls))
            @test !isfinite(M[k, k])
            ref = DampedLinearSolve.DenseLinearSystem{Float64}(M, copy(p.b), zeros(n))
            for mk in (() -> SecondOrderOptimization.MarquardtData(0.0, zeros(n)),
                       () -> SecondOrderOptimization.MarquardtDataTrial(0.0, zeros(n), nothing, NaN, false))
                lmd, lmd_ref = mk(), mk()
                SecondOrderOptimization.updatescaling!(lmd, ls)
                SecondOrderOptimization.updatescaling!(lmd_ref, ref)
                @test all(isfinite, lmd.scaling)
                @test lmd.scaling == lmd_ref.scaling
                @test lmd.lambda == lmd_ref.lambda
            end
        end
    end
end

# AS3: the dense Schur system's Bunch-Kaufman solve, for :ldl and as :auto's fallback, through the contract: the
# status, the retained part of the solution (the first nr entries, in the layout order of the vectors) against a
# factorization of the full matrix, and no allocation. (That it equals the solve of the reduced system as written
# before, Cholesky then LinearAlgebra's bunchkaufman, is BlockLinearSystems' test.) Only the retained part is
# compared, as before: after an Indefinite solve the eliminated part is not back-substituted.
function as3_reference(p, solver)
    Hs = Symmetric(p.H)
    if solver === :auto
        F = cholesky(Hs; check = false)
        issuccess(F) && return F \ p.b
    end
    return bunchkaufman(Hs) \ p.b
end
as3_solve_allocs(ls, s) = @allocated SecondOrderOptimization.solve!(ls, s)
as3_allocs(x, obj, o, w) = (x .= 0.1; @allocated optimize!(x, obj, o; workspace = w))

@testset "The dense Schur system's Bunch-Kaufman solve allocates nothing (AS3)" begin
    p = standard_problem()
    n = length(p.b)
    q = standard_problem()
    q.H[1, 1] = -50.0                        # S indefinite, so Cholesky fails (B5)
    zeroS(ls) = (getA(ls).R .= 0; getA(ls).C_data .= 0; ls)   # S = 0: Bunch-Kaufman finds an exactly zero pivot
    for (name, prob, mk, expected) in (("positive definite", p, s -> make_schur_ls(:dense, p; linsolver = s), Dict(:auto => :Success, :ldl => :Success)),
                                       ("indefinite", q, s -> make_schur_ls(:dense, q; linsolver = s), Dict(:auto => :Indefinite, :ldl => :Indefinite)),
                                       ("singular", nothing, s -> zeroS(make_schur_ls(:dense, p; linsolver = s)), Dict(:auto => :NotPositiveDefinite, :ldl => :NotPositiveDefinite)))
        for s in (:auto, :ldl)
            @testset "$name, $s" begin
                ls = mk(s)
                SecondOrderOptimization.dampen!(ls, 0.0)
                st = SecondOrderOptimization.solve!(ls, static(s))
                @test Symbol(st) === expected[s]
                if prob !== nothing
                    xr_ref = as3_reference(prob, s)[1:nr(prob)]
                    @test norm(getx(ls)[1:nr(prob)] - xr_ref) <= 1e-12 * norm(xr_ref)
                end
                # Warm: a solve allocates nothing (the system is serial here: one chunk)
                as3_solve_allocs(ls, static(s))
                @test as3_solve_allocs(ls, static(s)) == 0
            end
        end
    end
    # A whole Newton or dogleg run on an indefinite Schur problem allocates nothing (typed workspace)
    for it in (:newton, :dogleg), s in (:auto, :ldl)
        @testset "$it, $s run" begin
            cost, fdf! = as1_objective(q, dense_schur_matrix(q))
            obj = HessianObjective(cost, fdf!; H = dense_schur_matrix(q))
            opts = OptimizationOptions(; iterator = it, linsolver = s, maxiters = 30)
            ws = OptimizationWorkspace(fill(0.1, n), obj, opts)
            x = fill(0.1, n)
            as3_allocs(x, obj, opts, ws)
            @test as3_allocs(x, obj, opts, ws) == 0
        end
    end
end

