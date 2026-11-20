using SecondOrderOptimization, SparseArrays, LinearAlgebra, Test

# The solvers of the extensions (KrylovExt, PardisoExt: DampedLinearSolve's) through the options and the
# iterators. The solve-level tests are DampedLinearSolve's.
@testset "extensions.jl" begin
    # Validation of linsolver by OptimizationOptions, before loading extensions
    @test_throws ArgumentError OptimizationOptions(linsolver=:not_a_solver)
    # A2: there are no Schur solver symbols (the type of H selects elimination), and no
    # `reduction` option
    for s in (:schur, :dense_schur, :sparse_schur)
        err = try OptimizationOptions(linsolver=s); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("Unknown linear solver", sprint(showerror, err))
    end
    @test_throws MethodError OptimizationOptions(reduction=:eliminate)

    using Krylov
    # X2: :krylov was an alias of :cg and is gone; it fails validation like any unknown solver
    err = try OptimizationOptions(linsolver=:krylov); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("Unknown linear solver", sprint(showerror, err))

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
    for solver in (:cg, :minres)
        # Newton iterator
        opts_newton = OptimizationOptions(iterator=:newton, linsolver=solver, maxiters=10, gtol=1e-6)
        x_newton, res_newton = optimize([0.0, 0.0, 0.0], HessianObjective(quad_cost, quad_fdf!; H = copy(A_spd)), opts_newton)
        @test isconverged(res_newton)
        @test isapprox(x_newton, x_exact; rtol=1e-5)

        # Levenberg-Marquardt iterator
        opts_lm = OptimizationOptions(iterator=:levenbergmarquardt, linsolver=solver, maxiters=20, gtol=1e-6)
        x_lm, res_lm = optimize([0.0, 0.0, 0.0], HessianObjective(quad_cost, quad_fdf!; H = copy(A_spd)), opts_lm)
        @test isconverged(res_lm)
        @test isapprox(x_lm, x_exact; rtol=1e-5)
    end

    A_dense = Matrix(A_spd)
    quad_dense_cost(x) = 0.5 * dot(x, A_dense * x) - dot(b_vec, x)
    function quad_dense_fdf!(g, H, x)
        g .= A_dense * x .- b_vec
        H .= A_dense
        return 0.5 * dot(x, A_dense * x) - dot(b_vec, x)
    end

    for solver in (:cg, :minres)
        opts = OptimizationOptions(iterator=:newton, linsolver=solver, maxiters=10, gtol=1e-6)
        x_dense, res_dense = optimize([0.0, 0.0, 0.0], HessianObjective(quad_dense_cost, quad_dense_fdf!; H = copy(A_dense)), opts)
        @test isconverged(res_dense)
        @test isapprox(x_dense, x_exact; rtol=1e-5)
    end

    A_indef = sparse([-5.0 1.0 0.0; 1.0 6.0 1.0; 0.0 1.0 7.0])
    indef_cost(x) = 0.5 * dot(x, A_indef * x)
    function indef_fdf!(g, H, x)
        g .= A_indef * x
        H .= A_indef
        return 0.5 * dot(x, A_indef * x)
    end
    # With Newton, the indefinite Hessian makes CG return Indefinite, prompting Newton to
    # regularize with a diagonal shift
    opts_indef = OptimizationOptions(iterator=:newton, linsolver=:cg, maxiters=10)
    x_indef, res_indef = optimize([1.0, 1.0, 1.0], HessianObjective(indef_cost, indef_fdf!; H = copy(A_indef)), opts_indef)
    @test isfinite(res_indef.bestcost)

    using Pardiso
    if Pardiso.mkl_is_available() || Pardiso.panua_is_available()
        opts_pardiso = OptimizationOptions(iterator=:newton, linsolver=:pardiso, maxiters=10, gtol=1e-6)
        x_pardiso, res_pardiso = optimize([0.0, 0.0, 0.0], HessianObjective(quad_cost, quad_fdf!; H = copy(A_spd)), opts_pardiso)
        @test isconverged(res_pardiso)
        @test isapprox(x_pardiso, x_exact; rtol=1e-5)
        # Dense systems
        x_pd, res_pd = optimize([0.0, 0.0, 0.0], HessianObjective(quad_dense_cost, quad_dense_fdf!; H = copy(A_dense)), opts_pardiso)
        @test isconverged(res_pd)
        @test isapprox(x_pd, x_exact; rtol=1e-5)
        # Newton recovers from the indefinite Hessian by shifting
        x_pi, res_pi = optimize([1.0, 1.0, 1.0], HessianObjective(indef_cost, indef_fdf!; H = copy(A_indef)), OptimizationOptions(iterator=:newton, linsolver=:pardiso, maxiters=10))
        @test isfinite(res_pi.bestcost)
    end
end
