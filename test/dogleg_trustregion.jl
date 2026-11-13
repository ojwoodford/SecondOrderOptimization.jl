using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Test, Static

@testset "dogleg_trustregion.jl - Nocedal & Wright Dogleg Trust-Region" begin
    # 1. Quadratic problem: Newton step is exact minimizer, converges in 1 iteration
    quad_cost(x) = 0.5 * (x[1] - 3.0)^2 + 2.0 * (x[2] + 4.0)^2
    function quad_fdf!(g, H, x)
        g[1] = x[1] - 3.0
        g[2] = 4.0 * (x[2] + 4.0)
        H[1, 1] = 1.0
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 4.0
        return quad_cost(x)
    end

    x0_q = @SVector [0.0, 0.0]
    opts_q = OptimizationOptions(iterator=:dogleg, gtol=1e-10)
    x_opt_q, res_q = optimize(x0_q, HessianObjective(quad_cost, quad_fdf!), opts_q)
    @test isapprox(x_opt_q, [3.0, -4.0]; atol=1e-8)
    @test res_q.bestcost < 1e-16
    @test res_q.niterations == 1 # Newton-first fast path achieves 1-iteration convergence on quadratic!

    # 2. Dogleg boundary intersection verification
    # Setup problem where ||delta_N|| > Delta and ||delta_C|| < Delta
    # A = [10 0; 0 1], b = [10; 1] at x = [0, 0]
    # g = -b = [-10; -1]
    # delta_N = [1.0, 1.0], norm ~ 1.414
    # delta_C ~ [1.009, 0.1009], norm ~ 1.014
    # If Delta = 1.2: ||delta_C|| ~ 1.014 < 1.2 < ||delta_N|| ~ 1.414!
    function aniso_cost(x)
        return 5.0 * (x[1] - 1.0)^2 + 0.5 * (x[2] - 1.0)^2
    end
    function aniso_fdf!(g, H, x)
        g[1] = 10.0 * (x[1] - 1.0)
        g[2] = 1.0 * (x[2] - 1.0)
        H[1, 1] = 10.0
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 1.0
        return aniso_cost(x)
    end

    # Test dogleg step interpolation directly
    ls = SecondOrderOptimization.makelinearsystem(@SVector([0.0, 0.0]), nothing, :auto)
    dd = DoglegData(ls)
    dd.trustradius = 1.2 # Forces alpha (~1.014) < trustradius (1.2) < newton_norm (~1.414)
    data = OptimizationInternal(ls, @SVector([0.0, 0.0]), SecondOrderOptimization.Stats())
    cost0 = SecondOrderOptimization.computelinearization!(data, aniso_fdf!, data.variables)
    data.bestcost = cost0
    cost1 = SecondOrderOptimization.iterate!(dd, data, aniso_cost, aniso_fdf!, OptimizationOptions(iterator=:dogleg))
    step = SecondOrderOptimization.getx(data.linsystem)
    # The step must have norm exactly equal to trustradius (1.2) before potential expansion
    @test isapprox(norm(step), 1.2; atol=1e-6)
    # Cost must be lower
    @test cost1 < aniso_cost(@SVector([0.0, 0.0]))
    # Since rho is near 1 (> 0.75) and step was on boundary (>= 0.9 Delta), trustradius expanded to 2.4
    @test isapprox(dd.trustradius, 2.4; atol=1e-6)

    # 3. 1D Bratu ODE Boundary Value Problem: slashes iterations from 16 to <= 4
    function bratu_problem(n::Int; lambda=1.0)
        h = 1.0 / (n + 1)
        inv_h2 = 1.0 / (h * h)
        function f(u)
            res = zeros(n)
            res[1] = (2.0 * u[1] - (n > 1 ? u[2] : 0.0)) * inv_h2 - lambda * exp(u[1])
            for i in 2:n-1
                res[i] = (-u[i-1] + 2.0 * u[i] - u[i+1]) * inv_h2 - lambda * exp(u[i])
            end
            if n > 1
                res[n] = (-u[n-1] + 2.0 * u[n]) * inv_h2 - lambda * exp(u[n])
            end
            return 0.5 * sum(abs2, res)
        end
        function fdf!(g, H, u)
            res = zeros(n)
            res[1] = (2.0 * u[1] - (n > 1 ? u[2] : 0.0)) * inv_h2 - lambda * exp(u[1])
            for i in 2:n-1
                res[i] = (-u[i-1] + 2.0 * u[i] - u[i+1]) * inv_h2 - lambda * exp(u[i])
            end
            if n > 1
                res[n] = (-u[n-1] + 2.0 * u[n]) * inv_h2 - lambda * exp(u[n])
            end
            J = spzeros(n, n)
            J[1, 1] = 2.0 * inv_h2 - lambda * exp(u[1])
            if n > 1
                J[1, 2] = -inv_h2
            end
            for i in 2:n-1
                J[i, i-1] = -inv_h2
                J[i, i] = 2.0 * inv_h2 - lambda * exp(u[i])
                J[i, i+1] = -inv_h2
            end
            if n > 1
                J[n, n-1] = -inv_h2
                J[n, n] = 2.0 * inv_h2 - lambda * exp(u[n])
            end
            mul!(g, transpose(J), res)
            H_sp = transpose(J) * J
            if H isa SparseMatrixCSC
                copyto!(H, H_sp)
            else
                copyto!(H, Matrix(H_sp))
            end
            return 0.5 * sum(abs2, res)
        end
        return f, fdf!
    end

    f_bratu, fdf_bratu! = bratu_problem(64)
    x0_b = zeros(64)
    opts_b = OptimizationOptions(iterator=:dogleg, gtol=1e-8, maxiters=50)
    x_opt_b, res_b = optimize(copy(x0_b), HessianObjective(f_bratu, fdf_bratu!), opts_b)
    @test isconverged(res_b)
    @test res_b.niterations <= 4 # Slashed from 16 to 3-4 iterations!
    @test res_b.bestcost < 1e-20

    # 4. Zero allocations with static vectors and OptimizationWorkspace
    x0_stat = @SVector [1.5, -2.5]
    opts_stat = OptimizationOptions(iterator=:dogleg)
    ws = OptimizationWorkspace(x0_stat, HessianObjective(quad_cost, quad_fdf!), opts_stat)
    # Warmup
    optimize(x0_stat, HessianObjective(quad_cost, quad_fdf!), opts_stat; workspace=ws)
    # Measured inside a function: at testset top level Julia 1.10 boxes the captured variables
    dogleg_allocs(x, obj, o, w) = @allocated optimize(x, obj, o; workspace=w)
    dogleg_allocs(x0_stat, HessianObjective(quad_cost, quad_fdf!), opts_stat, ws)
    allocs = dogleg_allocs(x0_stat, HessianObjective(quad_cost, quad_fdf!), opts_stat, ws)
    @test allocs == 0

    # 5. DoglegData constructors and reset!
    ls_s = SecondOrderOptimization.makelinearsystem(x0_stat, nothing, :auto)
    dd1 = DoglegData(ls_s)
    @test dd1.trustradius == 0.0
    @test dd1.maxtrustradius == 1e10
    dd2 = DoglegData(ls_s, 100.0)
    @test dd2.maxtrustradius == 100.0
    dd1.trustradius = 50.0
    SecondOrderOptimization.reset!(dd1)
    @test dd1.trustradius == 0.0
end
