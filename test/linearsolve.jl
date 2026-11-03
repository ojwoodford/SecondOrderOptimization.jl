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
        SecondOrderOptimization.solve!(ls, static(solver))
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # StaticLinearSystem
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = SecondOrderOptimization.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
        SecondOrderOptimization.solve!(ls, static(solver))
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # SparseLinearSystem
    A_sp = sparse(A_spd)
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp), static(solver))
        ls.b .= b_vec
        SecondOrderOptimization.solve!(ls, static(solver))
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # 2. Indefinite symmetric system
    A_indef = randn(5, 5)
    A_indef = 0.5 * (A_indef + A_indef')
    # Make sure it's not positive definite
    A_indef[1, 1] -= 20.0
    A_indef[5, 5] += 20.0
    b_indef = A_indef * x_true

    # LDL on Dense
    ls_dense = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    SecondOrderOptimization.solve!(ls_dense, static(:ldl))
    @test isapprox(ls_dense.x, x_true; rtol=1e-8)

    # QR on Dense
    ls_dense_qr = SecondOrderOptimization.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    SecondOrderOptimization.solve!(ls_dense_qr, static(:qr))
    @test isapprox(ls_dense_qr.x, x_true; rtol=1e-8)

    # LDL on Sparse
    A_sp_indef = sparse(A_indef)
    ls_sp = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp_indef), static(:ldl))
    ls_sp.b .= b_indef
    SecondOrderOptimization.solve!(ls_sp, static(:ldl))
    @test isapprox(ls_sp.x, x_true; rtol=1e-8)

    # QR on Sparse
    ls_sp_qr = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp_indef), static(:qr))
    ls_sp_qr.b .= b_indef
    SecondOrderOptimization.solve!(ls_sp_qr, static(:qr))
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
    SecondOrderOptimization.try_cholesky!(x_out, copy(A_spd), b_vec)
    @test isapprox(x_out, x_true; rtol=1e-8)
    SecondOrderOptimization.try_cholesky!(x_out, copy(A_indef), b_indef)
    @test isapprox(x_out, x_true; rtol=1e-8)

    @test SecondOrderOptimization.max_diag(A_spd) ≈ maximum(diag(A_spd))
    ls_sp_test = SecondOrderOptimization.makelinearsystem(zeros(5), copy(A_sp), static(:ldl))
    @test SecondOrderOptimization.max_diag(ls_sp_test) ≈ maximum(diag(A_spd))

    A_damp = copy(A_spd)
    d = [A_damp[j, j] for j in 1:5]
    SecondOrderOptimization.damp_marquardt!(A_damp, d, 0.5)
    for j in 1:5
        @test A_damp[j, j] ≈ d[j] * 1.5
    end
end
