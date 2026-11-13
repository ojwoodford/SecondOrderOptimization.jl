using SecondOrderOptimization, StaticArrays, Static, Test
const SO = SecondOrderOptimization

# A4: one statement of the static-size rule
@testset "staticrule.jl" begin
    @test MAX_STATIC_SYSTEM == 16
    cases = ((1.0, true), (SVector(1.0), true), (zeros(SVector{16}), true), (zeros(SVector{17}), false),
             ((zeros(SVector{8}), zeros(SVector{8})), true), ((zeros(SVector{9}), zeros(SVector{8})), false),
             (SVector(SVector(1.0, 2.0), SVector(3.0, 4.0)), true), (zeros(5), false), (zeros(16), false))
    for (x0, expected) in cases
        @test usesstaticsystem(x0) == expected
        for s in (:auto, :ldl)
            ls = @inferred SO.makelinearsystem(x0, nothing, static(s))   # static, as OptimizationOptions passes it
            @test (ls isa SO.StaticLinearSystem) == usesstaticsystem(x0)
            @test length(SO.getx(ls)) == SO.dynamic(SO.nvars(x0))   # sized by nvars, not length
        end
    end
    # An SVector above the limit still optimizes, through a dense system
    x17, res17 = optimize(fill(2.0, SVector{17}), HessianObjective(x -> 0.5 * sum(abs2, x), (g, H, x) -> (g .= x; H .= 0; for i in 1:17; H[i, i] = 1.0; end; 0.5 * sum(abs2, x))))
    @test isconverged(res17)
    @test x17 isa SVector{17} && maximum(abs, x17) < 1e-8
end
