using SecondOrderOptimization, StaticArrays, LinearAlgebra, Static, Test
const SOO = SecondOrderOptimization

# collectstats = false (item 15): the same run, with every time and allocation field of the result zero

cs_f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
function cs_fdf!(g, H, x)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    H[1, 1] = 2 - 400 * x[2] + 1200 * x[1]^2
    H[1, 2] = H[2, 1] = -400 * x[1]
    H[2, 2] = 200
    return cs_f(x)
end
cs_res(x) = (1 - x[1], 10 * (x[2] - x[1]^2))
cs_ls_f(x) = 0.5 * sum(abs2, cs_res(x))
function cs_resjac!(r, J, x)
    r[1], r[2] = cs_res(x)
    J[1, 1] = -1.0; J[1, 2] = 0.0
    J[2, 1] = -20 * x[1]; J[2, 2] = 10.0
    return cs_ls_f(x)
end
const CS_OBJECTIVES = (
    ("Hessian, static", () -> SVector(-1.2, 1.0), HessianObjective(cs_f, cs_fdf!)),
    ("Hessian, dense", () -> [-1.2, 1.0], HessianObjective(cs_f, cs_fdf!; H = zeros(2, 2))),
    ("Jacobian, static", () -> SVector(-1.2, 1.0), JacobianObjective(cs_ls_f, cs_resjac!, 2)),
    ("Jacobian, dense", () -> [-1.2, 1.0], JacobianObjective(cs_ls_f, cs_resjac!, zeros(2, 2))),
)
const CS_ZERO_FIELDS = (:timetotal, :timeinit, :timecost, :timegradient, :timesolver,
    :initallocations, :initallocated, :optimizeallocations, :optimizeallocated)
const CS_KEPT_FIELDS = (:startcost, :bestcost, :niterations, :costcomputations, :gradientcomputations, :linearsolves,
    :termination, :iterator)

cs_warm(x, obj, opts, ws) = @allocated optimize!(x, obj, opts; workspace = ws)

@testset "collectstats = false: $name, :$it" for (name, mkx, obj) in CS_OBJECTIVES,
        it in (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg)
    on = OptimizationOptions(iterator = it)
    off = OptimizationOptions(iterator = it, collectstats = false)
    @test on.collectstats === True() && off.collectstats === False()
    xon, ron = optimize(mkx(), obj, on)
    xoff, roff = optimize(mkx(), obj, off)
    # The same run
    @test isequal(xon, xoff)
    for field in CS_KEPT_FIELDS
        @test isequal(getfield(ron, field), getfield(roff, field))
    end
    # Its statistics are zero, and were collected with the flag on
    for field in CS_ZERO_FIELDS
        @test getfield(roff, field) == 0
    end
    @test ron.timetotal > 0
    # show says so, and divides by nothing
    s = sprint(show, roff)
    @test occursin("Statistics were not collected (collectstats = false)", s) && !occursin("NaN", s)
    @test !occursin("Statistics were not collected", sprint(show, ron))
    # Inferred, and nothing allocates after warm-up
    @test (@inferred optimize(mkx(), obj, off)) isa Tuple
    if mkx() isa Vector
        ws = OptimizationWorkspace(mkx(), obj, off)
        cs_warm(mkx(), obj, off, ws)
        @test cs_warm(mkx(), obj, off, ws) == 0
        # A workspace built for one setting is not reused for the other
        @test !SOO.workspacevalid(ws, mkx(), on, SOO.objective_structure(obj))
        @test SOO.workspacevalid(ws, mkx(), off, SOO.objective_structure(obj))
    end
end

@testset "collectstats = false: maxtime still stops a run, and CostTrajectory's times increase" begin
    slow(x) = (sleep(0.002); cs_f(x))
    slow!(g, H, x) = (sleep(0.002); cs_fdf!(g, H, x))
    _, r = optimize([-1.2, 1.0], HessianObjective(slow, slow!), OptimizationOptions(iterator = :levenbergmarquardt, collectstats = false, maxtime = 0.01))
    @test r.termination & TERMINATION_MAX_TIME != 0
    @test r.timetotal == 0
    ct = CostTrajectory()
    optimize([-1.2, 1.0], HessianObjective(cs_f, cs_fdf!), OptimizationOptions(collectstats = false); callback = storecostscallback(ct))
    @test length(ct.times_ns) > 1 && all(>(0), ct.times_ns) && issorted(ct.times_ns) && allunique(ct.times_ns)
end

@testset "OptimizationInternal's three-argument constructor collects statistics" begin
    ls = SOO.makehessiansystem(zeros(2, 2), :auto)
    @test SOO.collectsstats(SOO.OptimizationInternal(ls, [0.0, 0.0], SOO.Stats())) === True()
    @test SOO.collectsstats(SOO.OptimizationInternal(ls, [0.0, 0.0], SOO.Stats(), False())) === False()
end
