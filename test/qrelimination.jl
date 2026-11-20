using SecondOrderOptimization, BlockLinearSystems, Krylov, SparseArrays, LinearAlgebra, Static, Random, Test
using Polyester   # BlockLinearSystems' kernels thread only with Polyester loaded
using SecondOrderOptimization: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!
include("fixtures/load.jl"); loadfixtures("tp3.jl")
const SOO = SecondOrderOptimization

@testset "qrelimination.jl" begin

    @testset "TP6: :qr on a sparse reduced system equals the dense one's, under-determined block = $under" for under in (false, true)
        p = TP3Problem(tp3_structure(; under))
        x = 0.1 .* randn(Xoshiro(2), p.lay.n)
        sc = 0.5 .+ rand(Xoshiro(3), p.lay.n)
        lsD = tp3_system(p, Matrix{Float64}, :qr, x)
        lsP = tp3_system(p, SparseMatrixCSC{Float64, Int}, :qr, x)
        for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, sc), (1e-6, sc))
            aD, aP = tp3_solve!(lsD, :qr, lambda, scaling), tp3_solve!(lsP, :qr, lambda, scaling)
            @test aD == aP
            aD == Success || continue
            @test getx(lsP) ≈ getx(lsD) rtol = 1e-12
        end
        # (That the sparse reduced system's J̃ and r̃ are the dense path's, entry for entry, and that filling them
        # allocates nothing, is BlockLinearSystems' test: they are its reduced system's internals.)
        # SPQR allocates the same per solve whatever the damping; nothing else does (rules.md 21's exception)
        k = tp6_allocations(lsP, 0.3, sc)
        @test k > 0
        @test tp6_allocations(lsP, 1e-3, sc) == tp6_allocations(lsP, 3.0, nothing) == k
        # a warm run with a workspace allocates exactly SPQR's allocations for each solve
        for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton), ccs in (false, true)
            under && continue
            opts = OptimizationOptions(; iterator = it, linsolver = :qr, callbackchangesstate = ccs)
            obj = tp3_objective(p, SparseMatrixCSC{Float64, Int})
            ws = OptimizationWorkspace(fill(0.1, p.lay.n), obj, opts)
            xw = fill(0.1, p.lay.n)
            optimize!(xw, obj, opts; workspace = ws)
            GC.gc()
            xw .= 0.1
            res = optimize!(xw, obj, opts; workspace = ws)
            lsw = ws.data[1].linsystem
            @test isconverged(res) && res.linearsolves > 0
            @test res.optimizeallocations == res.linearsolves * tp6_allocations(ws.data[1].linsystem, 0.3, nothing)
        end
    end


    @testset "Every Jacobian iterator: $RT, :$s, :$it, callbackchangesstate = $ccs" for RT in keys(TP3_SOLVERS), s in TP3_SOLVERS[RT], it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton), ccs in (false, true)
        # Every (prototype, solver, iterator) at ccs = false, the path that builds a trial system with
        # copy_structure and swaps it with the live one; ccs = true, which uses a subset of those methods,
        # with :auto only, for each prototype and iterator
        ccs && s !== :auto && continue
        p = TP3Problem(tp3_structure())
        opts = OptimizationOptions(; iterator = it, linsolver = s, callbackchangesstate = ccs, maxiters = 200)
        _, rd = optimize(fill(0.1, p.lay.n), tp3_dense_objective(p), OptimizationOptions(; iterator = it, callbackchangesstate = ccs, maxiters = 200))
        obj = tp3_objective(p, RT)
        x, res = optimize(fill(0.1, p.lay.n), obj, opts)
        @test isconverged(res)
        @test res.bestcost ≈ rd.bestcost rtol = 1e-8
        # a warm run with a workspace: no allocation, except CHOLMOD's working memory in the sparse Cholesky
        # and SPQR's in the sparse QR (the stated exception; TP6's testset counts SPQR's)
        if RT <: SparseMatrixCSC && s in (:auto, :qr, :cholesky)
            @test tp3_warm_bytes(p, RT, opts) > 0
        else
            @test tp3_warm_bytes(p, RT, opts) == 0
        end
    end

    @testset "Invalid prototypes and (RType, linsolver) pairs throw up front" begin
        p = TP3Problem(tp3_structure())
        called = Ref(false)
        f(x) = (called[] = true; tp3_cost(p, x))
        rj!(r, J, x) = (called[] = true; tp3_tables!(p, J, x))
        x0 = fill(0.1, p.lay.n)
        for (RT, s) in ((Matrix{Float64}, :cg), (Matrix{Float64}, :minres),
                        (SparseMatrixCSC{Float64, Int}, :cg))
            obj = JacobianObjective(f, rj!, SchurBlockJacobian{Float64, RT}(p.jst))
            err = try optimize(x0, obj, OptimizationOptions(linsolver = s)); nothing catch e; e end
            @test err isa ArgumentError && occursin("SchurBlockJacobian", err.msg) && occursin(":$s", err.msg)
            @test_throws ArgumentError OptimizationWorkspace(x0, obj, OptimizationOptions(linsolver = s))
        end
        @test !called[]
        @test_throws ArgumentError JacobianObjective(f, rj!, SchurBlockJacobian{Float32, Matrix{Float32}}(p.jst))
        @test_throws DimensionMismatch optimize(zeros(3), JacobianObjective(f, rj!, SchurBlockJacobian{Float64, Matrix{Float64}}(p.jst)))
        @test !called[]
        # The message shared by every Jacobian-path error names the solvers each SchurBlockJacobian takes
        msg = sprint(showerror, try JacobianObjective(f, rj!, SchurBlockJacobian{Float32, Matrix{Float32}}(p.jst)); nothing catch e; e end)
        @test occursin("SchurBlockJacobian{Float64, Matrix{Float64}} or SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}, linsolver = :auto, :qr, :cholesky or :ldl", msg)
        @test !called[]
    end

    @testset "Workspaces: reused for the same structure, rebuilt for another" begin
        p = TP3Problem(tp3_structure())
        opts = OptimizationOptions()
        ws = OptimizationWorkspace(fill(0.1, p.lay.n), tp3_objective(p, Matrix{Float64}), opts)
        data = ws.data[1]        # (the loop may swap the main and trial systems, so compare the data)
        x, res = optimize(fill(0.1, p.lay.n), tp3_objective(p, Matrix{Float64}), opts; workspace = ws)
        @test ws.data[1] === data && isconverged(res)
        p2 = TP3Problem(tp3_structure(; under = true))
        wsany = OptimizationWorkspace()
        optimize(fill(0.1, p.lay.n), tp3_objective(p, Matrix{Float64}), opts; workspace = wsany)
        data1 = wsany.data[1]
        optimize(fill(0.1, p2.lay.n), tp3_objective(p2, Matrix{Float64}), opts; workspace = wsany)
        @test wsany.data[1] !== data1
    end
    @testset "LM with :lsmr takes :qr's iterations on a badly scaled problem (LP1): $RT, :$it" for RT in keys(TP3_SOLVERS), it in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        p = tp3_scaled_problem()
        runs = Dict(s => optimize(fill(0.1, p.lay.n), tp3_objective(p, RT), OptimizationOptions(; iterator = it, linsolver = s, maxiters = 200))[2] for s in (:qr, :lsmr))
        @test isconverged(runs[:qr]) && isconverged(runs[:lsmr])
        @test abs(runs[:lsmr].niterations - runs[:qr].niterations) <= 1
        @test runs[:lsmr].bestcost <= runs[:qr].bestcost * (1 + 1e-8) + 1e-20
    end
end
