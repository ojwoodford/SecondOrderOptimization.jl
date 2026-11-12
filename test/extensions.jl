!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using SecondOrderOptimization, SparseArrays, LinearAlgebra, Test, Static

@testset "extensions.jl" begin
    # 1. Validation error before loading extensions
    # (Testing that unregistered solver symbols throw informative ArgumentErrors)
    @test_throws ArgumentError OptimizationOptions(linsolver=:not_a_solver)
    # B9: the Schur solvers need BlockLinearSystems, and the error says so
    if Base.get_extension(SecondOrderOptimization, :BlockLinearSystemsExt) === nothing
        for s in (:schur, :dense_schur, :sparse_schur)
            @test !SecondOrderOptimization.validatelinsolver(s)
            err = try OptimizationOptions(linsolver=s); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("BlockLinearSystems", sprint(showerror, err))
        end
    end

    # 2. KrylovExt testing
    using Krylov

    @test SecondOrderOptimization.validatelinsolver(:cg)
    @test SecondOrderOptimization.validatelinsolver(:minres)
    @test SecondOrderOptimization.validatelinsolver(:lsmr)
    @test SecondOrderOptimization.validatelinsolver(:krylov)

    # Quadratic test problem: f(x) = 0.5 * x' A x - b' x
    A_spd = sparse([5.0 1.0 0.0; 1.0 6.0 1.0; 0.0 1.0 7.0])
    b_vec = [1.0, 2.0, 3.0]
    x_exact = A_spd \ b_vec

    quad_cost(x) = 0.5 * dot(x, A_spd * x) - dot(b_vec, x)
    function quad_fdf!(g, H, x)
        g .= A_spd * x .- b_vec
        H .= A_spd
        return 0.5 * dot(x, A_spd * x) - dot(b_vec, x)
    end

    # Test Krylov solvers on SparseLinearSystem
    for solver in (:cg, :minres, :lsmr, :krylov)
        # Newton iterator
        opts_newton = OptimizationOptions(iterator=:newton, linsolver=solver, maxiters=10, gtol=1e-6)
        x_newton, res_newton = optimize(quad_cost, quad_fdf!, [0.0, 0.0, 0.0], opts_newton; H=copy(A_spd))
        @test isconverged(res_newton)
        @test isapprox(x_newton, x_exact; rtol=1e-5)

        # Levenberg-Marquardt iterator
        opts_lm = OptimizationOptions(iterator=:levenbergmarquardt, linsolver=solver, maxiters=20, gtol=1e-6)
        x_lm, res_lm = optimize(quad_cost, quad_fdf!, [0.0, 0.0, 0.0], opts_lm; H=copy(A_spd))
        @test isconverged(res_lm)
        @test isapprox(x_lm, x_exact; rtol=1e-5)
    end

    # B1: a sparse system built for another solver throws an informative error rather than
    # trying to replace the `const` workspace field
    for solver in (:cg, :minres, :lsmr)
        ls_wrong = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_spd), static(:auto))
        ls_wrong.b .= b_vec
        @test_throws ArgumentError SecondOrderOptimization.solve!(ls_wrong, static(solver))
        ls_right = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_spd), static(solver))
        ls_right.b .= b_vec
        @test SecondOrderOptimization.solve!(ls_right, static(solver)) == Success
        @test isapprox(ls_right.x, x_exact; rtol=1e-5)
    end

    # Test Krylov on DenseLinearSystem
    A_dense = Matrix(A_spd)
    quad_dense_cost(x) = 0.5 * dot(x, A_dense * x) - dot(b_vec, x)
    function quad_dense_fdf!(g, H, x)
        g .= A_dense * x .- b_vec
        H .= A_dense
        return 0.5 * dot(x, A_dense * x) - dot(b_vec, x)
    end

    for solver in (:cg, :minres, :lsmr)
        # M9: dense Krylov solves use a preallocated workspace
        ls_dk = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_dense), static(solver))
        @test ls_dk.solver_ws isa Krylov.KrylovWorkspace
        ls_dk.b .= b_vec
        @test SecondOrderOptimization.solve!(ls_dk, static(solver)) == Success
        @test isapprox(ls_dk.x, x_exact; rtol=1e-5)
        dense_krylov_allocs(ls, s) = @allocated SecondOrderOptimization.solve!(ls, s)
        dense_krylov_allocs(ls_dk, static(solver))
        @test dense_krylov_allocs(ls_dk, static(solver)) == 0
        opts = OptimizationOptions(iterator=:newton, linsolver=solver, maxiters=10, gtol=1e-6)
        x_dense, res_dense = optimize(quad_dense_cost, quad_dense_fdf!, [0.0, 0.0, 0.0], opts; H=copy(A_dense))
        @test isconverged(res_dense)
        @test isapprox(x_dense, x_exact; rtol=1e-5)
    end

    # Test Krylov negative curvature handling
    A_indef = sparse([-5.0 1.0 0.0; 1.0 6.0 1.0; 0.0 1.0 7.0])
    indef_cost(x) = 0.5 * dot(x, A_indef * x)
    function indef_fdf!(g, H, x)
        g .= A_indef * x
        H .= A_indef
        return 0.5 * dot(x, A_indef * x)
    end
    # B4: CG reports nonpositive curvature as Indefinite, on both sparse and dense systems
    for H_indef in (copy(A_indef), Matrix(A_indef))
        ls_indef = SecondOrderOptimization.makelinearsystem(zeros(3), H_indef, static(:cg))
        ls_indef.b .= [1.0, 0.0, 0.0]
        @test SecondOrderOptimization.solve!(ls_indef, static(:cg)) == Indefinite
        # Damping enough to make it positive definite gives Success
        dampen!(ls_indef, 10.0)
        @test SecondOrderOptimization.solve!(ls_indef, static(:cg)) == Success
        @test isapprox(ls_indef.x, (Matrix(A_indef) + 10I) \ [1.0, 0.0, 0.0]; rtol=1e-6)
    end
    # With Newton, the indefinite Hessian makes CG return Indefinite, prompting Newton to
    # regularize with a diagonal shift
    opts_indef = OptimizationOptions(iterator=:newton, linsolver=:cg, maxiters=10)
    x_indef, res_indef = optimize(indef_cost, indef_fdf!, [1.0, 1.0, 1.0], opts_indef; H=copy(A_indef))
    @test isfinite(res_indef.bestcost)

    # 3. PardisoExt testing
    using Pardiso

    @test SecondOrderOptimization.validatelinsolver(:pardiso)

    if Pardiso.mkl_is_available() || Pardiso.panua_is_available()
        opts_pardiso = OptimizationOptions(iterator=:newton, linsolver=:pardiso, maxiters=10, gtol=1e-6)
        x_pardiso, res_pardiso = optimize(quad_cost, quad_fdf!, [0.0, 0.0, 0.0], opts_pardiso; H=copy(A_spd))
        @test isconverged(res_pardiso)
        @test isapprox(x_pardiso, x_exact; rtol=1e-5)
        # B1: repeated solves on one system must not try to reassign the const workspace field
        ls = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_spd), static(:pardiso))
        ls.b .= b_vec
        for _ in 1:3
            @test SecondOrderOptimization.solve!(ls, static(:pardiso)) == Success
            @test isapprox(ls.x, x_exact; rtol=1e-8)
        end
        ls_wrong = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_spd), static(:auto))
        @test_throws ArgumentError SecondOrderOptimization.solve!(ls_wrong, static(:pardiso))
        # B8: damping changes values only, so the analysed pattern is reused
        dampen!(ls, 2.0)
        @test SecondOrderOptimization.solve!(ls, static(:pardiso)) == Success
        @test isapprox(ls.x, (Matrix(A_spd) + 2I) \ b_vec; rtol=1e-8)
        @test ls.A == A_spd   # damping undone
        dampen!(ls, 0.0)
        # B8: an indefinite matrix is reported, not silently solved as indefinite
        for H_indef in (copy(A_indef), Matrix(A_indef))
            ls_indef = SecondOrderOptimization.makelinearsystem(zeros(3), H_indef, static(:pardiso))
            ls_indef.b .= b_vec
            @test SecondOrderOptimization.solve!(ls_indef, static(:pardiso)) == NotPositiveDefinite
            dampen!(ls_indef, 10.0)
            @test SecondOrderOptimization.solve!(ls_indef, static(:pardiso)) == Success
            @test isapprox(ls_indef.x, (Matrix(A_indef) + 10I) \ b_vec; rtol=1e-8)
        end
        # Dense systems
        x_pd, res_pd = optimize(quad_dense_cost, quad_dense_fdf!, [0.0, 0.0, 0.0], opts_pardiso; H=copy(A_dense))
        @test isconverged(res_pd)
        @test isapprox(x_pd, x_exact; rtol=1e-5)
        # Newton recovers from the indefinite Hessian by shifting
        x_pi, res_pi = optimize(indef_cost, indef_fdf!, [1.0, 1.0, 1.0], OptimizationOptions(iterator=:newton, linsolver=:pardiso, maxiters=10); H=copy(A_indef))
        @test isfinite(res_pi.bestcost)
    else
        # When shared library is not installed, attempting to solve throws an informative error
        ls = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_spd), static(:pardiso))
        @test_throws ErrorException SecondOrderOptimization.solve!(ls, static(:pardiso))
        ls_dense = SecondOrderOptimization.makelinearsystem(zeros(3), Matrix(A_spd), static(:pardiso))
        @test_throws ErrorException SecondOrderOptimization.solve!(ls_dense, static(:pardiso))
    end

    # B8: the lower triangle handed to Pardiso, and its map back into the full matrix
    PExt = Base.get_extension(SecondOrderOptimization, :PardisoExt)
    A_full = sparse([4.0 1.0 0.0 2.0; 1.0 5.0 3.0 0.0; 0.0 3.0 6.0 0.0; 2.0 0.0 0.0 7.0])
    L, lmap = PExt.lower_triangle_with_map(A_full, Int32)
    @test L isa SparseMatrixCSC{Float64, Int32}
    @test L == tril(A_full)
    @test L.nzval == A_full.nzval[lmap]
end
