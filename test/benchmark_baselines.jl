# Baseline profiling and allocation benchmark script across Static, Dense, and Sparse paths
using SecondOrderOptimization
using SecondOrderOptimization: StaticLinearSystem, DenseLinearSystem, makelinearsystem
using StaticArrays, SparseArrays, LinearAlgebra, Printf, Static

println("================================================================================")
println("SecondOrderOptimization.jl Linear Solver Baselines & Allocation Audit")
println("================================================================================")

# 1. Static Linear Systems (D = 3, 6)
println("\n--- 1. Static Linear System Benchmarks ---")
for D in (3, 6)
    A_mat = randn(D, D)
    A_mat = A_mat * A_mat' + 2.0 * I
    A_static = MMatrix{D, D, Float64}(A_mat)
    b_static = MVector{D, Float64}(randn(D))
    x_static = MVector{D, Float64}(zeros(D))
    ls_static = StaticLinearSystem{D, D*D, Float64}(A_static, b_static, x_static)

    # Warmup
    for s in (:auto, :cholesky, :ldl, :qr)
        solve!(ls_static, s)
    end

    for s in (:auto, :cholesky, :ldl, :qr)
        sym = static(s)
        alloc = @allocated solve!(ls_static, sym)
        # timing
        t0 = time_ns()
        N_reps = 10_000
        for _ in 1:N_reps
            solve!(ls_static, sym)
        end
        dt_ns = (time_ns() - t0) / N_reps
        @printf("Static D=%-2d | Solver: %-10s | Allocations: %5d bytes | Time: %8.2f ns\n", D, string(s), alloc, dt_ns)
    end
end

# 2. Dense Linear Systems (N = 50, 100)
println("\n--- 2. Dense Linear System Benchmarks ---")
for N in (50, 100)
    A_mat = randn(N, N)
    A_mat = A_mat * A_mat' + 2.0 * I
    b_vec = randn(N)
    x_vec = zeros(N)
    ls_dense = DenseLinearSystem(copy(A_mat), copy(b_vec), copy(x_vec))

    # Warmup
    for s in (:auto, :cholesky, :ldl, :qr)
        solve!(ls_dense, s)
    end

    for s in (:auto, :cholesky, :ldl, :qr)
        sym = static(s)
        alloc = @allocated solve!(ls_dense, sym)
        t0 = time_ns()
        N_reps = 200
        for _ in 1:N_reps
            solve!(ls_dense, sym)
        end
        dt_us = (time_ns() - t0) / (N_reps * 1000)
        @printf("Dense N=%-3d | Solver: %-10s | Allocations: %5d bytes | Time: %8.2f μs\n", N, string(s), alloc, dt_us)
    end
end

# 3. Sparse Linear Systems (N = 500)
println("\n--- 3. Sparse Linear System Benchmarks ---")
begin
    N = 500
    S = sprandn(N, N, 0.05)
    H_sparse = S * S' + 5.0 * I
    b_sparse = randn(N)
    x_sparse = zeros(N)

    # Auto / LDL
    ls_ldl = makelinearsystem(x_sparse, H_sparse, :ldl)
    solve!(ls_ldl, :ldl)
    alloc_ldl = @allocated solve!(ls_ldl, :ldl)
    t0 = time_ns()
    N_reps = 50
    for _ in 1:N_reps
        solve!(ls_ldl, :ldl)
    end
    dt_ldl_ms = (time_ns() - t0) / (N_reps * 1_000_000)
    @printf("Sparse N=%-3d | Solver: %-10s | Allocations: %5d bytes | Time: %8.2f ms\n", N, "ldl", alloc_ldl, dt_ldl_ms)

    # Cholesky (cached factor)
    ls_chol = makelinearsystem(x_sparse, H_sparse, :cholesky)
    solve!(ls_chol, :cholesky)
    alloc_chol = @allocated solve!(ls_chol, :cholesky)
    t0 = time_ns()
    for _ in 1:N_reps
        solve!(ls_chol, :cholesky)
    end
    dt_chol_ms = (time_ns() - t0) / (N_reps * 1_000_000)
    @printf("Sparse N=%-3d | Solver: %-10s | Allocations: %5d bytes | Time: %8.2f ms\n", N, "cholesky", alloc_chol, dt_chol_ms)

    # QR (SuiteSparse SPQR)
    ls_qr = makelinearsystem(x_sparse, H_sparse, :qr)
    solve!(ls_qr, :qr)
    alloc_qr = @allocated solve!(ls_qr, :qr)
    t0 = time_ns()
    N_reps_qr = 10
    for _ in 1:N_reps_qr
        solve!(ls_qr, :qr)
    end
    dt_qr_ms = (time_ns() - t0) / (N_reps_qr * 1_000_000)
    @printf("Sparse N=%-3d | Solver: %-10s | Allocations: %5d bytes | Time: %8.2f ms\n", N, "qr", alloc_qr, dt_qr_ms)
end

println("\n================================================================================")
println("✓ Baseline profiling benchmark completed.")
println("================================================================================")
