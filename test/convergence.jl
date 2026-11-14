using SecondOrderOptimization, StaticArrays, LinearAlgebra, Static, Test
include("termbits.jl")
const SOO = SecondOrderOptimization

# Q6: what counts as converged (isconverged), and the two-sided cost-change tests of optimizeloop!

# A termination value from masks, and isconverged on it
q6conv(masks::Integer...) = SOO.isconverged_bits(reduce(|, masks; init=0))

const Q6_CONVERGENCE = (TERMINATION_REL_DCOST, TERMINATION_ABS_DCOST, TERMINATION_STEP_SIZE, TERMINATION_GRAD_TOL, TERMINATION_STALLED)   # STALLED: Q7
const Q6_LIMITS = (TERMINATION_MAX_FAILS, TERMINATION_MAX_ITERS, TERMINATION_MAX_TIME)
const Q6_NONFINITE = (TERMINATION_COST_INF, TERMINATION_COST_NAN, TERMINATION_STEP_INF, TERMINATION_STEP_NAN, TERMINATION_GRAD_NOT_FINITE, TERMINATION_START_NOT_FINITE)

@testset "convergence.jl" begin
    @testset "isconverged on hand-built termination values" begin
        # Q8: termination_reasons is gone; the masks and isconverged are the public interface
        @test !isdefined(SOO, :termination_reasons) && !isdefined(@__MODULE__, :termination_reasons)
        @test all(b -> isdefined(SOO, b) && b in names(SOO), (:TERMINATION_REL_DCOST, :TERMINATION_MAX_FAILS, :TERMINATION_GRAD_TOL, :isconverged))
        @test all(x -> length(x) == 2 && x[2] isa String, SOO.TERMINATION_REASONS)    # bit and message only
        # the rules below cover every bit that has a message
        @test Set(x[1] for x in SOO.TERMINATION_REASONS) == Set([Q6_CONVERGENCE..., Q6_LIMITS..., Q6_NONFINITE..., TERMINATION_COST_DISCREPANCY, TERMINATION_LINSOLVER_FAILED])
        # a convergence bit alone is converged
        for c in Q6_CONVERGENCE
            @test q6conv(c)
        end
        # a convergence bit with a limit is converged; a limit alone is not
        for c in Q6_CONVERGENCE, l in Q6_LIMITS
            @test q6conv(c, l)
        end
        for l in Q6_LIMITS
            @test !q6conv(l)
        end
        @test q6conv(TERMINATION_REL_DCOST, TERMINATION_ABS_DCOST, TERMINATION_MAX_FAILS, TERMINATION_MAX_ITERS, TERMINATION_MAX_TIME)
        # grad_tol with cost_discrepancy is converged; cost_discrepancy alone is not
        @test q6conv(TERMINATION_GRAD_TOL, TERMINATION_COST_DISCREPANCY)
        @test !q6conv(TERMINATION_COST_DISCREPANCY)
        # anything non-finite is a failure whatever else is set
        for nf in Q6_NONFINITE
            @test !q6conv(nf)
            for c in Q6_CONVERGENCE
                @test !q6conv(c, nf)
                @test !q6conv(c, nf, TERMINATION_MAX_ITERS)
            end
        end
        @test !q6conv(TERMINATION_STEP_SIZE, TERMINATION_COST_NAN) && !q6conv(TERMINATION_STEP_SIZE, TERMINATION_COST_INF)        # a small step to a non-finite cost
        @test !q6conv(TERMINATION_GRAD_TOL, TERMINATION_START_NOT_FINITE, TERMINATION_COST_INF)                          # a small gradient at a start with no finite cost
        # linsolver_failed is a failure
        @test !q6conv(TERMINATION_LINSOLVER_FAILED)
        for c in Q6_CONVERGENCE
            @test !q6conv(c, TERMINATION_LINSOLVER_FAILED)
        end
        # nothing set, or only a callback's flags, is not converged
        @test !SOO.isconverged_bits(0)
        @test !SOO.isconverged_bits(1 << 16)
    end

    # The two-residual Rosenbrock, as a Jacobian objective
    function rosen_resjac!(r, J, x)
        r[1] = 1 - x[1]; r[2] = 10 * (x[2] - x[1]^2)
        J[1, 1] = -1.0; J[2, 1] = -20x[1]; J[2, 2] = 10.0
        return 0.5 * (r[1]^2 + r[2]^2)
    end
    rosen = JacobianObjective(x -> 0.5 * ((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2), rosen_resjac!, 2)
    # An exponential fit with a nonzero residual at its minimum
    tq = collect(range(0, 2; length=25)); yq = @. 3exp(-1.3tq) + 0.05sin(7tq)
    q6resid(x) = @. x[1] * exp(x[2] * tq) - yq
    expfit = JacobianObjective(x -> 0.5 * sum(abs2, q6resid(x)),
                               (r, J, x) -> (r .= q6resid(x); J[:, 1] .= exp.(x[2] .* tq); J[:, 2] .= x[1] .* tq .* exp.(x[2] .* tq); 0.5 * sum(abs2, r)), 25)

    @testset "Constructed runs, one per rule" begin
        # a run that meets a convergence criterion on its last allowed iteration is converged.
        # (grad_tol is tested after a relinearization, which a max_iters stop never reaches, so the
        # criterion here is the change in cost.)
        opts(; kw...) = OptimizationOptions(; reldcost=1e-10, absdcost=0.0, dstep=0.0, kw...)
        _, res = optimize([1.0, 0.0], expfit, opts())
        @test isconverged(res) && res.termination == TERMINATION_REL_DCOST
        n = res.niterations
        _, res = optimize([1.0, 0.0], expfit, opts(maxiters=n - 1))
        @test res.termination == TERMINATION_MAX_ITERS && !isconverged(res)              # max_iters alone
        _, res = optimize([1.0, 0.0], expfit, opts(maxiters=n))
        @test res.termination == TERMINATION_REL_DCOST | TERMINATION_MAX_ITERS
        @test isconverged(res)                                                              # a limit with a convergence bit
        # grad_tol alone
        _, res = optimize([-1.2, 1.0], rosen, OptimizationOptions(gtol=1e-6, reldcost=0.0, absdcost=0.0, dstep=0.0))
        @test res.termination == TERMINATION_GRAD_TOL && isconverged(res)
        # a small gradient with a cost discrepancy: fdf! reports a cost 1e-6 above f away from the start
        f(x) = 0.5 * (x[1] - 1)^2
        objd = HessianObjective(f, (g, H, x) -> (g[1] = x[1] - 1; H[1, 1] = 1.0; f(x) + (x[1] == 0.0 ? 0.0 : 1e-6)))
        # (callbackchangesstate = true: the plain Newton variant scores the step with f, the
        # relinearizing one with fdf!, which would hide the discrepancy)
        _, res = optimize([0.0], objd, OptimizationOptions(iterator=:newton, gtol=1e-10, callbackchangesstate=true))
        @test res.termination == TERMINATION_COST_DISCREPANCY | TERMINATION_GRAD_TOL
        @test isconverged(res)
        # a small gradient at a start with no finite cost (converged before Q6)
        objinf = HessianObjective(x -> Inf, (g, H, x) -> (g .= 0.0; H .= I(1); Inf))
        _, res = optimize([0.0], objinf, OptimizationOptions(gtol=1e-3))
        @test hasbits(res, TERMINATION_GRAD_TOL) && hasbits(res, TERMINATION_START_NOT_FINITE)
        @test !isconverged(res)
        # a small step to a point whose cost is not finite: the cost is NaN away from the start, and
        # every trial is rejected until the step is below dstep
        fn(x) = x[1] == 0.0 ? 0.5 * (x[1] - 1)^2 : NaN
        objn = HessianObjective(fn, (g, H, x) -> (g[1] = x[1] - 1; H[1, 1] = 1.0; fn(x)))
        _, res = optimize([0.0], objn, OptimizationOptions(iterator=:dogleg, dstep=1e-3))
        @test hasbits(res, TERMINATION_COST_NAN) && !isconverged(res)
        # a linear-solver failure: an indefinite Hessian that :cholesky cannot solve, even shifted
        objf = HessianObjective(x -> -0.5 * x[1]^2 - x[1], (g, H, x) -> (g[1] = -x[1] - 1; H[1, 1] = -1e300; -0.5 * x[1]^2 - x[1]); H=zeros(1, 1))
        _, res = optimize([0.0], objf, OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:cholesky))
        @test hasbits(res, TERMINATION_LINSOLVER_FAILED) && !isconverged(res)
    end

    @testset "Gauss-Newton converges at the minimum (it ended on max_fails before Q6)" begin
        # a well-conditioned fit: Gauss-Newton accepts every step, and at the minimum the cost moves
        # up and down by rounding; a rise below the tolerance now ends the run as converged
        t = collect(range(0, 2; length=25)); y = @. 3exp(-1.3t) + 0.05sin(7t)
        resid(x) = @. x[1] * exp(x[2] * t) - y
        costs = Float64[]
        obj = JacobianObjective(x -> 0.5 * sum(abs2, resid(x)),
                                (r, J, x) -> (r .= resid(x); J[:, 1] .= exp.(x[2] .* t); J[:, 2] .= x[1] .* t .* exp.(x[2] .* t); 0.5 * sum(abs2, r)), 25)
        x, res = optimize([1.0, 0.0], obj, OptimizationOptions(iterator=:newton); callback=(c, args...) -> (push!(costs, c); (c, 0)))
        @test isconverged(res)
        @test hasbits(res, TERMINATION_REL_DCOST | TERMINATION_ABS_DCOST)
        # the point returned is the best one, also when the last step was a small rise
        @test res.bestcost == minimum([res.startcost; costs])
        @test 0.5 * sum(abs2, resid(x)) == res.bestcost
        # a rise above the tolerance is still a failure, and a run that keeps rising still ends on
        # max_fails, not converged: Newton on √(1 + x²) from |x| > 1 maps x to -x³
        fd(x) = sqrt(1 + x[1]^2)
        objdiv = HessianObjective(fd, (g, H, x) -> (s = fd(x); g[1] = x[1] / s; H[1, 1] = 1 / s^3; s))
        xd, res = optimize([1.5], objdiv, OptimizationOptions(iterator=:newton, maxiters=20))
        @test res.termination == TERMINATION_MAX_FAILS
        @test !isconverged(res)
        @test res.niterations == 4 && xd == [1.5]            # maxstall = 3, and the start is the best point
    end
end

# A variable whose update shrinks chosen steps by 1e-9, so that an iteration makes a tiny (not
# significant) improvement on cue: `pattern(k)` for the k-th update says whether to shrink it
struct Stutter
    v::SVector{2, Float64}
end
const STUTTER_CALLS = Ref(0)
const STUTTER_PATTERN = Ref{Function}(k -> false)
SOO.nvars(::Stutter) = static(2)
function SOO.update(x::Stutter, dx, start=1)
    STUTTER_CALLS[] += 1
    s = STUTTER_PATTERN[](STUTTER_CALLS[]) ? 1e-9 : 1.0
    return Stutter(x.v + s * SVector(dx[start], dx[start + 1]))
end

# A variable that ignores the step and walks through a scripted list of costs, to drive the
# stall counter exactly: the k-th update moves to cost SCRIPT[][k + 1]
struct Scripted
    k::Int
end
const SCRIPT = Ref([1.0])
SOO.nvars(::Scripted) = static(1)
SOO.update(x::Scripted, dx, start=1) = Scripted(x.k + 1)
scost(x::Scripted) = SCRIPT[][min(x.k + 1, end)]
const SCRIPTED_OBJ = JacobianObjective(scost, (r, J, x::Scripted) -> (c = scost(x); r[1] = sqrt(2c); J[1, 1] = 1.0; c), 1)

@testset "Q7: a tight per-iteration test, and a stall test merged with maxstall" begin
    @test OptimizationOptions().reldcost == 1e-12 && OptimizationOptions().stalltol == 1.5e-8
    @test OptimizationOptions().absdcost == 1e-15 && OptimizationOptions().dstep == 1e-15 && OptimizationOptions().maxstall == 3
    @test OptimizationOptions(stalltol=0).stalltol === 0.0
    # maxstall replaces maxfails (Q7: it counts iterations without a significant improvement)
    @test OptimizationOptions(maxstall=7).maxstall == 7
    @test_throws MethodError OptimizationOptions(maxfails=3)
    @test !hasfield(typeof(OptimizationOptions()), :maxfails)
    @test TERMINATION_STALLED == 1 << 15 && TERMINATION_STALLED in [x[1] for x in SOO.TERMINATION_REASONS]

    @testset "isconverged with the stalled bit" begin
        @test q6conv(TERMINATION_STALLED)
        for l in Q6_LIMITS
            @test q6conv(TERMINATION_STALLED, l)
        end
        for nf in Q6_NONFINITE
            @test !q6conv(TERMINATION_STALLED, nf)
        end
        @test !q6conv(TERMINATION_STALLED, TERMINATION_LINSOLVER_FAILED)
    end

    # The exponential fit (well conditioned) and the degree-13 polynomial fit (ill conditioned)
    t = collect(range(0, 2; length=25)); y = @. 3exp(-1.3t) + 0.05sin(7t)
    eresid(x) = @. x[1] * exp(x[2] * t) - y
    function eresjac!(r, J, x)
        r .= eresid(x)
        J[:, 1] .= exp.(x[2] .* t); J[:, 2] .= x[1] .* t .* exp.(x[2] .* t)
        return 0.5 * sum(abs2, r)
    end
    expfit = JacobianObjective(x -> 0.5 * sum(abs2, eresid(x)), eresjac!, 25)

    @testset "Gauss-Newton: rel_dcost at once when well conditioned, stalled at the minimum when not" begin
        _, res = optimize([1.0, 0.0], expfit, OptimizationOptions(iterator=:newton))
        @test isconverged(res) && hasbits(res, TERMINATION_REL_DCOST) && !hasbits(res, TERMINATION_STALLED)
        tt = collect(range(0, 1; length=60))
        V = 1e4 * hcat((tt .^ k for k in 0:13)...)
        yv = 1e4 * (@. sin(3tt) + 0.01 * cos(17tt))
        exactcost(x) = Float64(0.5 * sum(abs2, big.(V) * big.(x) - big.(yv)))
        costmin = exactcost(big.(V) \ big.(yv))
        obj = JacobianObjective(x -> 0.5 * sum(abs2, V * x - yv), (r, J, x) -> (r .= V * x - yv; J .= V; 0.5 * sum(abs2, r)), 60)
        for ccs in (false, true)
            x, res = optimize(zeros(14), obj, OptimizationOptions(iterator=:newton, callbackchangesstate=ccs, maxiters=500))
            # the cost there is computable to about 1e-8 relative: Gauss-Newton bounces within that
            @test hasbits(res, TERMINATION_STALLED) && isconverged(res)
            @test !hasbits(res, TERMINATION_MAX_FAILS)
            @test (exactcost(x) - costmin) / costmin <= 1e-12
            @test res.niterations < 20
            # with stalltol = 0 (maxstall then behaves as maxfails did before Q7) it ends on max_fails alone, not converged
            _, res0 = optimize(zeros(14), obj, OptimizationOptions(iterator=:newton, callbackchangesstate=ccs, maxiters=500, stalltol=0))
            @test res0.termination == TERMINATION_MAX_FAILS && !isconverged(res0)
        end
    end

    @testset "A crawling run stalls: progress in bursts, with long dips" begin
        # uniform damping on Rosenbrock with x[2] in units of 1e-6. It averages 7e-5 relative per
        # iteration, but in bursts: over iterations 23-26 the best cost improves by 1.9e-9 in all, so
        # by the stall rule it has stalled, far from the minimum (0). Decided 2026-11-14: keep the
        # rule (a stall cannot tell a minimum from a valley crossed slowly). stalltol = 0 keeps it going.
        d = [1.0, 1e-6]
        f(x) = 0.5 * ((1 - x[1] * d[1])^2 + 100 * (x[2] * d[2] - (x[1] * d[1])^2)^2)
        objs = JacobianObjective(f, (r, J, x) -> (r[1] = 1 - x[1] * d[1]; r[2] = 10 * (x[2] * d[2] - (x[1] * d[1])^2);
                                                  J[1, 1] = -d[1]; J[2, 1] = -20 * x[1] * d[1]^2; J[2, 2] = 10 * d[2]; f(x)), 2)
        x0 = [-1.2, 1.0] ./ d
        _, res = optimize(x0, objs, OptimizationOptions(maxiters=500))
        @test res.termination == TERMINATION_STALLED && isconverged(res) && res.bestcost > 0.5
        _, res0 = optimize(x0, objs, OptimizationOptions(maxiters=500, stalltol=0.0))
        @test res0.termination == TERMINATION_MAX_ITERS && res0.bestcost < res.bestcost
        # the right iterator for it converges
        _, res3 = optimize(x0, objs, OptimizationOptions(iterator=:levenbergmarquardt_dtd, maxiters=500))
        @test isconverged(res3) && res3.bestcost < 1e-20
    end

    @testset "The counter resets on a significant improvement" begin
        # every other iteration makes a tiny improvement (its step shrunk by 1e-9): with maxstall = 1,
        # no two in a row are insignificant, so the run goes on to the minimum
        f(x::Stutter) = 0.5 * sum(abs2, eresid(x.v))
        sobj = JacobianObjective(f, (r, J, x::Stutter) -> eresjac!(r, J, x.v), 25)
        # (reldcost = absdcost = 0: near the minimum a shrunk step changes the cost by less than
        # 1e-12 relative, and the per-iteration test would stop the run first)
        opts(; kw...) = OptimizationOptions(; iterator=:newton, maxiters=100, reldcost=0.0, absdcost=0.0, kw...)
        _, ref = optimize([1.0, 0.0], expfit, opts())
        STUTTER_CALLS[] = 0; STUTTER_PATTERN[] = isodd
        xs, res = optimize(Stutter(SVector(1.0, 0.0)), sobj, opts(maxstall=1))
        @test isconverged(res)
        @test res.bestcost ≈ ref.bestcost rtol=1e-12
        # without the reset it would stop at iteration 3 (two insignificant iterations, 1 and 3)
        @test res.niterations > 3
        # with maxstall = 0 the first tiny improvement is too many: stalled at once
        STUTTER_CALLS[] = 0
        _, res0 = optimize(Stutter(SVector(1.0, 0.0)), sobj, opts(maxstall=0))
        @test res0.niterations == 1 && res0.termination == TERMINATION_STALLED
    end

    @testset "A run that stalls far from its minimum ends stalled (what the flag means)" begin
        # every step shrunk by 1e-9: the best cost improves negligibly, far from the minimum. That is
        # stalled, and reported converged: the flag says progress stopped, not that this is a minimum
        f(x::Stutter) = 0.5 * sum(abs2, eresid(x.v))
        sobj = JacobianObjective(f, (r, J, x::Stutter) -> eresjac!(r, J, x.v), 25)
        STUTTER_CALLS[] = 0; STUTTER_PATTERN[] = k -> true
        _, res = optimize(Stutter(SVector(1.0, 0.0)), sobj, OptimizationOptions(iterator=:newton))
        @test res.termination == TERMINATION_STALLED && isconverged(res)
        @test res.niterations == OptimizationOptions().maxstall + 1
        @test res.bestcost > 100 * 0.0144779      # the minimum is 0.0144779
        STUTTER_PATTERN[] = k -> false
    end

    @testset "Stalled or max_fails: the lowest cost since the last significant improvement decides" begin
        opts(; kw...) = OptimizationOptions(; iterator=:newton, maxiters=50, kw...)
        # one insignificant improvement, then rises beyond the band but below sqrt(stalltol) (1.2e-4):
        # the last cost is outside the band, but one iteration since the reset stayed within it, and
        # no rise is large, so the cost has settled: stalled, and the best point is returned
        SCRIPT[] = [1.0, 1.0 - 1e-10, 1.0 + 1e-5, 1.0 + 1e-5, 1.0 + 1e-5, 1.0 + 1e-5, 1.0 + 1e-5]
        x, res = optimize(Scripted(0), SCRIPTED_OBJ, opts())
        @test res.termination == TERMINATION_STALLED && res.niterations == 4 && isconverged(res)
        @test res.bestcost == 1.0 - 1e-10 && x == Scripted(1)
        # the same with rises of 1e-3, above sqrt(stalltol): the cost jumped, so not settled (a run
        # diverging after one insignificant improvement is not reported converged)
        SCRIPT[] = [1.0, 1.0 - 1e-10, 1.0 + 1e-3, 1.0 + 1e-3, 1.0 + 1e-3, 1.0 + 1e-3, 1.0 + 1e-3]
        x, res = optimize(Scripted(0), SCRIPTED_OBJ, opts())
        @test res.termination == TERMINATION_MAX_FAILS && res.niterations == 4 && !isconverged(res)
        @test res.bestcost == 1.0 - 1e-10 && x == Scripted(1)        # the best point is still returned
        # every cost a rise beyond the band: max_fails, not converged
        SCRIPT[] = [1.0, 1.0 + 1e-3, 1.0 + 2e-3, 1.0 + 3e-3, 1.0 + 4e-3, 1.0 + 5e-3]
        x, res = optimize(Scripted(0), SCRIPTED_OBJ, opts())
        @test res.termination == TERMINATION_MAX_FAILS && res.niterations == 4 && !isconverged(res)
        @test x == Scripted(0) && res.bestcost == 1.0
        # rises within the band count as settled too
        SCRIPT[] = [1.0, 1.0 + 1e-9, 1.0 + 1e-5, 1.0 + 1e-5, 1.0 + 1e-5, 1.0 + 1e-5]
        _, res = optimize(Scripted(0), SCRIPTED_OBJ, opts())
        @test res.termination == TERMINATION_STALLED
        # the rise bound is sqrt(stalltol), set when the options are created
        @test OptimizationOptions().stallrise == sqrt(1.5e-8) && OptimizationOptions(stalltol=1e-6).stallrise == 1e-3
        @test OptimizationOptions(stalltol=0).stallrise == 0.0
        SCRIPT[] = [1.0, 1.0 - 1e-10, 1.0 + 5e-4, 1.0 + 5e-4, 1.0 + 5e-4, 1.0 + 5e-4]
        @test optimize(Scripted(0), SCRIPTED_OBJ, opts())[2].termination == TERMINATION_MAX_FAILS                  # 5e-4 > 1.2e-4
        @test optimize(Scripted(0), SCRIPTED_OBJ, opts(stalltol=1e-6))[2].termination == TERMINATION_STALLED       # 5e-4 < 1e-3
        # stalltol = 0: the insignificant improvement is an improvement, which resets the counter,
        # and the run then ends on max_fails after maxstall + 1 rises, as before Q7
        SCRIPT[] = [1.0, 1.0 - 1e-10, 1.0 + 1e-3, 1.0 + 1e-3, 1.0 + 1e-3, 1.0 + 1e-3, 1.0 + 1e-3]
        _, res = optimize(Scripted(0), SCRIPTED_OBJ, opts(stalltol=0.0))
        @test res.termination == TERMINATION_MAX_FAILS && res.niterations == 5
        SCRIPT[] = [1.0]
    end

    @testset "A diverging run ends on max_fails, not stalled" begin
        fd(x) = sqrt(1 + x[1]^2)
        objdiv = HessianObjective(fd, (g, H, x) -> (s = fd(x); g[1] = x[1] / s; H[1, 1] = 1 / s^3; s))
        for st in (0.0, 1.5e-8, 0.5)
            _, res = optimize([1.5], objdiv, OptimizationOptions(iterator=:newton, maxiters=20, stalltol=st))
            @test res.termination == TERMINATION_MAX_FAILS && !isconverged(res)
        end
    end

    @testset "stalltol = 0: maxstall counts consecutive rises only" begin
        # many tiny improvements in a row never stop it
        f(x::Stutter) = 0.5 * sum(abs2, eresid(x.v))
        sobj = JacobianObjective(f, (r, J, x::Stutter) -> eresjac!(r, J, x.v), 25)
        STUTTER_CALLS[] = 0; STUTTER_PATTERN[] = k -> k <= 10
        _, res = optimize(Stutter(SVector(1.0, 0.0)), sobj, OptimizationOptions(iterator=:newton, stalltol=0.0, reldcost=0.0, absdcost=0.0))
        @test res.niterations > 10
        @test isapprox(res.bestcost, 0.014477931372569955; rtol=1e-12) && !hasbits(res, TERMINATION_STALLED)
        STUTTER_PATTERN[] = k -> false
    end
end
