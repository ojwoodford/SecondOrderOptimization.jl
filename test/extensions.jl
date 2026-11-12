!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using SecondOrderOptimization, SparseArrays, LinearAlgebra, Test, Static

@testset "extensions.jl" begin
    # 1. Validation error before loading extensions
    # (Testing that unregistered solver symbols throw informative ArgumentErrors)
    @test_throws ArgumentError OptimizationOptions(linsolver=:not_a_solver)

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
    # With Newton, indefinite Hessian causes CG to detect negative curvature, returning Indefinite,
    # prompting Newton to regularize with spectral shift
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
    else
        # When shared library is not installed, attempting to solve throws an informative error
        ls = SecondOrderOptimization.makelinearsystem(zeros(3), copy(A_spd), static(:pardiso))
        @test_throws ErrorException SecondOrderOptimization.solve!(ls, static(:pardiso))
    end
end
