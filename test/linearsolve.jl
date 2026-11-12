using SecondOrderOptimization, SparseArrays, StaticArrays, LinearAlgebra, Test, Static

@testset "linearsolve.jl" begin
    # 1. Symmetric positive definite system
    A_mat = randn(5, 5)
    A_spd = A_mat' * A_mat + 5.0 * I
    x_true = randn(5)
    b_vec = A_spd * x_true

    # DenseLinearSystem
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_spd), copy(b_vec), zeros(5))
        status = SecondOrderOptimization.solve!(ls, static(solver))
        @test status == SecondOrderOptimization.Success
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # StaticLinearSystem
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
        status = SecondOrderOptimization.solve!(ls, static(solver))
        @test status == SecondOrderOptimization.Success
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # SparseLinearSystem
    A_sp = sparse(A_spd)
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp), static(solver))
        ls.b .= b_vec
        status = SecondOrderOptimization.solve!(ls, static(solver))
        @test status == SecondOrderOptimization.Success
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # 2. Indefinite symmetric system
    A_indef = randn(5, 5)
    A_indef = 0.5 * (A_indef + A_indef')
    # Make sure it's not positive definite
    A_indef[1, 1] -= 20.0
    A_indef[5, 5] += 20.0
    b_indef = A_indef * x_true

    # LDL on Dense (returns Indefinite, still solves)
    ls_dense = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    status_dense_ldl = SecondOrderOptimization.solve!(ls_dense, static(:ldl))
    @test status_dense_ldl == SecondOrderOptimization.Indefinite
    @test isapprox(ls_dense.x, x_true; rtol=1e-8)

    # Cholesky on Dense with indefinite system fails cleanly without fallback
    ls_dense_chol = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    status_dense_chol = SecondOrderOptimization.solve!(ls_dense_chol, static(:cholesky))
    @test status_dense_chol == SecondOrderOptimization.NotPositiveDefinite

    # Cholesky on Static with indefinite system fails cleanly without fallback
    ls_static_chol = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_indef), MVector{5}(b_indef), zeros(MVector{5, Float64}))
    status_static_chol = SecondOrderOptimization.solve!(ls_static_chol, static(:cholesky))
    @test status_static_chol == SecondOrderOptimization.NotPositiveDefinite

    # QR on Dense
    ls_dense_qr = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    status_dense_qr = SecondOrderOptimization.solve!(ls_dense_qr, static(:qr))
    @test status_dense_qr == SecondOrderOptimization.Success
    @test isapprox(ls_dense_qr.x, x_true; rtol=1e-8)

    # LDL on Sparse (returns Indefinite, still solves)
    A_sp_indef = sparse(A_indef)
    ls_sp = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp_indef), static(:ldl))
    ls_sp.b .= b_indef
    status_sp_ldl = SecondOrderOptimization.solve!(ls_sp, static(:ldl))
    @test status_sp_ldl == SecondOrderOptimization.Indefinite
    @test isapprox(ls_sp.x, x_true; rtol=1e-8)

    # Cholesky on Sparse with indefinite system fails cleanly without fallback
    ls_sp_chol = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp_indef), static(:cholesky))
    ls_sp_chol.b .= b_indef
    status_sp_chol = SecondOrderOptimization.solve!(ls_sp_chol, static(:cholesky))
    @test status_sp_chol == SecondOrderOptimization.NotPositiveDefinite

    # QR on Sparse
    ls_sp_qr = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp_indef), static(:qr))
    ls_sp_qr.b .= b_indef
    status_sp_qr = SecondOrderOptimization.solve!(ls_sp_qr, static(:qr))
    @test status_sp_qr == SecondOrderOptimization.Success
    @test isapprox(ls_sp_qr.x, x_true; rtol=1e-8)

    # 3. invsym (dense and static symmetric matrices)
    for N in (2, 3)
        M = randn(N, N)
        S = M' * M + 2.0 * I
        @test isapprox(SecondOrderOptimization.invsym(S), inv(S); rtol=1e-12)
        S_inv = SecondOrderOptimization.invsym(SMatrix{N, N, Float64}(S))
        @test S_inv isa SMatrix{N, N, Float64}
        @test isapprox(Matrix(S_inv), inv(S); rtol=1e-12)
    end

    # 4. try_cholesky!, max_diag, damp_marquardt!, uniformscaling!
    x_out = zeros(5)
    @test SecondOrderOptimization.try_cholesky!(x_out, copy(A_spd), b_vec) == SecondOrderOptimization.Success
    @test isapprox(x_out, x_true; rtol=1e-8)
    @test SecondOrderOptimization.try_cholesky!(x_out, copy(A_indef), b_indef) == SecondOrderOptimization.NotPositiveDefinite

    @test SecondOrderOptimization.max_diag(A_spd) ≈ maximum(diag(A_spd))
    ls_sp_test = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp), static(:ldl))
    @test SecondOrderOptimization.max_diag(ls_sp_test) ≈ maximum(diag(A_spd))

    A_damp = copy(A_spd)
    d = [A_damp[j, j] for j in 1:5]
    SecondOrderOptimization.damp_marquardt!(A_damp, d, 0.5)
    for j in 1:5
        @test A_damp[j, j] ≈ d[j] * 1.5
    end

    # 5. Damping contract: zero mutation on dense/static, exact restoration on sparse, no drift
    ls_d = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_spd), copy(b_vec), zeros(5))
    SecondOrderOptimization.dampen!(ls_d, 3.0)
    @test ls_d.lambda == 3.0
    SecondOrderOptimization.solve!(ls_d, static(:cholesky))
    x_damped_true = (A_spd + 3.0 * I) \ b_vec
    @test isapprox(ls_d.x, x_damped_true; rtol=1e-8)
    @test ls_d.A == A_spd # base matrix never mutated!

    ls_s = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
    SecondOrderOptimization.dampen!(ls_s, 3.0)
    SecondOrderOptimization.solve!(ls_s, static(:cholesky))
    @test isapprox(ls_s.x, x_damped_true; rtol=1e-8)
    @test Matrix(ls_s.A) == A_spd # base matrix never mutated!

    ls_sparse = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp), static(:ldl))
    ls_sparse.b .= b_vec
    SecondOrderOptimization.dampen!(ls_sparse, 3.0)
    SecondOrderOptimization.solve!(ls_sparse, static(:ldl))
    @test isapprox(ls_sparse.x, x_damped_true; rtol=1e-8)
    @test ls_sparse.A.nzval[ls_sparse.diag_inds] == diag(A_spd)

    # Sparse CHOLMOD native shift test
    ls_chol_sp = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp), static(:cholesky))
    ls_chol_sp.b .= b_vec
    SecondOrderOptimization.dampen!(ls_chol_sp, 3.0)
    SecondOrderOptimization.solve!(ls_chol_sp, static(:cholesky))
    @test isapprox(ls_chol_sp.x, x_damped_true; rtol=1e-8)
    @test ls_chol_sp.A.nzval[ls_chol_sp.diag_inds] == diag(A_spd)

    # 6. Linear solver failure termination signaling via LINSOLVER_FAILED
    unsolvable_cost(x) = x[1]^2 + x[2]^2
    function unsolvable_fdf!(g, H, x)
        g[1] = 2.0 * x[1]
        g[2] = 2.0 * x[2]
        H[1, 1] = 0.0
        H[2, 2] = 0.0
        H[1, 2] = 1e20
        H[2, 1] = 1e20
        return x[1]^2 + x[2]^2
    end

    for iter in (:newton, :levenbergmarquardt)
        opts = OptimizationOptions(iterator=iter, linsolver=:cholesky, maxiters=10)
        x0 = @SVector [1.0, 1.0]
        x_opt, res = optimize(unsolvable_cost, unsolvable_fdf!, x0, opts)
        @test (res.termination & TERMINATION_LINSOLVER_FAILED) != 0
        @test !isconverged(res)
    end

    # B6: a sparse prototype with missing diagonal entries keeps its index type
    for Ti in (Int32, Int64)
        H_missing = SparseMatrixCSC{Float64, Ti}(sparse(Ti[1, 2, 3, 1], Ti[1, 2, 1, 3], [4.0, 5.0, 1.0, 1.0], 3, 3))
        @test nnz(H_missing) == 4   # (3, 3) is missing
        A_fixed, dinds = SecondOrderOptimization.ensure_diag_and_inds(H_missing)
        @test A_fixed isa SparseMatrixCSC{Float64, Ti}
        @test eltype(dinds) == Ti
        @test all(A_fixed.rowval[dinds[c]] == c for c in 1:3)
        @test A_fixed == H_missing
        ls = SecondOrderOptimization.makelinearsystem(zeros(3), H_missing, static(:auto))
        @test ls isa SecondOrderOptimization.SparseLinearSystem{Float64, Ti}
        ls.A[3, 3] = 6.0
        ls.b .= [1.0, 2.0, 3.0]
        @test SecondOrderOptimization.solve!(ls, static(:auto)) == SecondOrderOptimization.Success
        @test isapprox(ls.x, Matrix(ls.A) \ [1.0, 2.0, 3.0]; rtol=1e-10)
    end

    # B10: the first Newton shift is finite even when the diagonal contains NaN
    ls_nan = SecondOrderOptimization.DenseLinearSystem{Float64}([NaN 0.0; 0.0 1.0], zeros(2), zeros(2))
    @test SecondOrderOptimization.initial_shift(ls_nan) == 1e-6
    ls_big = SecondOrderOptimization.DenseLinearSystem{Float64}([1e4 0.0; 0.0 1.0], zeros(2), zeros(2))
    @test SecondOrderOptimization.initial_shift(ls_big) == 1.0

    # M2: static :ldl solves an indefinite system and reports it, like dense and sparse :ldl
    ls_static_ldl = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_indef), MVector{5}(b_indef), zeros(MVector{5, Float64}))
    @test SecondOrderOptimization.solve!(ls_static_ldl, static(:ldl)) == SecondOrderOptimization.Indefinite
    @test isapprox(ls_static_ldl.x, x_true; rtol=1e-8)
    ls_static_ldl_pd = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
    @test SecondOrderOptimization.solve!(ls_static_ldl_pd, static(:ldl)) == SecondOrderOptimization.Success
    @test isapprox(ls_static_ldl_pd.x, x_true; rtol=1e-8)
    ls_static_sing = SecondOrderOptimization.StaticLinearSystem{2, 4, Float64}(MMatrix{2, 2}([1.0 1.0; 1.0 1.0]), MVector(1.0, 0.0), zeros(MVector{2, Float64}))
    @test SecondOrderOptimization.solve!(ls_static_sing, static(:ldl)) == SecondOrderOptimization.NotPositiveDefinite
    static_ldl_allocs(ls) = @allocated SecondOrderOptimization.solve!(ls, static(:ldl))
    static_ldl_allocs(ls_static_ldl)
    @test static_ldl_allocs(ls_static_ldl) == 0

    # M3: indefiniteness is detected relative to the scale of the matrix
    for scale in (1e-9, 1.0, 1e9)
        A_sc = scale .* A_indef
        b_sc = A_sc * x_true
        ls_d = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_sc), copy(b_sc), zeros(5))
        @test SecondOrderOptimization.solve!(ls_d, static(:ldl)) == SecondOrderOptimization.Indefinite
        ls_s = SecondOrderOptimization.makelinearsystem(zeros(5), sparse(A_sc), static(:ldl))
        ls_s.b .= b_sc
        @test SecondOrderOptimization.solve!(ls_s, static(:ldl)) == SecondOrderOptimization.Indefinite
        ls_st = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_sc), MVector{5}(b_sc), zeros(MVector{5, Float64}))
        @test SecondOrderOptimization.solve!(ls_st, static(:ldl)) == SecondOrderOptimization.Indefinite
        # and a positive definite matrix at the same scale is not flagged
        ls_p = SecondOrderOptimization.DenseLinearSystem{Float64}(scale .* A_spd, scale .* b_vec, zeros(5))
        @test SecondOrderOptimization.solve!(ls_p, static(:ldl)) == SecondOrderOptimization.Success
    end
    # A 2x2 Bunch-Kaufman pivot block with one negative eigenvalue is flagged
    bk2 = bunchkaufman(Symmetric([0.0 1.0; 1.0 0.0]))
    @test SecondOrderOptimization.is_indefinite(bk2)
    @test !SecondOrderOptimization.is_indefinite(bunchkaufman(Symmetric([2.0 1.0; 1.0 2.0])))

    # B2/M21: the indefiniteness check on a dense :ldl factorization costs O(n), not O(n^2), memory
    n_big = 300
    M_big = randn(n_big, n_big)
    A_big = M_big + M_big'
    bk_big = bunchkaufman(Symmetric(A_big))
    SecondOrderOptimization.is_indefinite(bk_big)
    @test (@allocated SecondOrderOptimization.is_indefinite(bk_big)) < 8 * n_big * n_big ÷ 10
    @test SecondOrderOptimization.is_indefinite(bk_big) == (eigmin(Symmetric(A_big)) < 0)
end
