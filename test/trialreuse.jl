using SecondOrderOptimization, StaticArrays, Test, Static

@testset "trialreuse.jl" begin
    # Exponential fitting problem: y_i = a * exp(b * x_i)
    # Track calls to f (cost only) and fdf! (cost + grad + hess)
    f_counts = Ref(0)
    fdf_counts = Ref(0)

    xs = range(0.0, 2.0, length=20)
    ys = [2.0 * exp(-1.3 * x) for x in xs]

    function exp_cost(p)
        f_counts[] += 1
        a, b = p[1], p[2]
        c = 0.0
        for (x, y) in zip(xs, ys)
            pred = a * exp(b * x)
            c += 0.5 * (pred - y)^2
        end
        return c
    end

    function exp_fdf!(g, H, p)
        fdf_counts[] += 1
        a, b = p[1], p[2]
        c = 0.0
        g[1] = 0.0
        g[2] = 0.0
        H[1, 1] = 0.0
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 0.0
        for (x, y) in zip(xs, ys)
            e = exp(b * x)
            pred = a * e
            r = pred - y
            c += 0.5 * r^2
            ja = e
            jb = a * x * e
            g[1] += ja * r
            g[2] += jb * r
            H[1, 1] += ja * ja
            H[1, 2] += ja * jb
            H[2, 2] += jb * jb
        end
        H[2, 1] = H[1, 2]
        return c
    end

    for it in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        f_counts[] = 0
        fdf_counts[] = 0
        p0 = @SVector [1.0, -0.5]
        opts = OptimizationOptions(iterator=it, maxiters=30, reldcost=1e-12)
        p_opt, res = optimize(p0, HessianObjective(exp_cost, exp_fdf!), opts)

        @test isapprox(p_opt, [2.0, -1.3]; atol=1e-3)
        # Because trial linearization is reused on accepted first trials,
        # fdf_counts[] should equal niterations + 1, and f_counts[] should be 0 (or very small on failed steps)
        @test fdf_counts[] == res.niterations + 1
        @test res.bestcost < 1e-8
    end
end
