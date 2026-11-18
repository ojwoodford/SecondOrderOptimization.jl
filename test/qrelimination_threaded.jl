using SecondOrderOptimization, BlockLinearSystems, Krylov, SparseArrays, LinearAlgebra, Static, Random, Test
using Polyester   # BlockLinearSystems' kernels thread only with Polyester loaded
using SecondOrderOptimization: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!
include("fixtures/load.jl"); loadfixtures("tp3.jl")
const SOO = SecondOrderOptimization

    @testset "Threaded elimination: the same steps; allocation" begin
        @test Base.get_extension(BlockLinearSystems, :PolyesterExt) !== nothing
        begin
            # One chunk per thread and no work floor, so that on more than one thread the kernels run threaded
            # (rules.md 30)
            p = TP3Problem(tp3_structure(; nchunks = Threads.nthreads(), min_work = 0))
            x = 0.1 .* randn(Xoshiro(2), p.lay.n)
            lsS, Jd, rd = tp3_schur(p, x)
            # Every threaded run once before any is measured: @allocated counts every thread, and Polyester's idle
            # worker threads fill Julia's method caches in the scheduler the first times they meet a new task,
            # which is the runtime's allocation, not a run's (the threaded Schur tests that used to run first, and
            # now are BlockLinearSystems', did this before)
            for RT in keys(TP3_SOLVERS), s in TP3_SOLVERS[RT], it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)
                tp3_warm_bytes(p, RT, OptimizationOptions(; iterator = it, linsolver = s))
            end
            for RT in keys(TP3_SOLVERS), s in TP3_SOLVERS[RT]
                ls = tp3_system(p, RT, s, x)
                @test tp3_solve!(ls, s, 0.3, nothing) == Success == tp3_solve!(lsS, :cholesky, 0.3, nothing)
                @test getx(ls) ≈ getx(lsS) rtol = s === :lsmr ? 1e-6 : 1e-10
                RT <: SparseMatrixCSC && s in (:auto, :qr, :cholesky) && continue   # CHOLMOD's or SPQR's working memory
                for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)
                    @test tp3_threaded_run_ok(tp3_warm_bytes(p, RT, OptimizationOptions(; iterator = it, linsolver = s)))
                end
            end
        end
    end
