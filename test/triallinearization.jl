using SecondOrderOptimization, SparseArrays, LinearAlgebra, Static, Test
using DampedLinearSolve: getA, getb, getx, copy_structure
const SOO = SecondOrderOptimization
const II = SecondOrderOptimization.IteratorInterface

# The shared trial linearization (src/triallinearization.jl), directly: with and without a spare system, ready and
# not ready, and the swap that the default linearize! makes

tl_f(x) = 0.5 * ((x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2)
function tl_fdf!(g, H, x)
    g[1] = (x[1] - 1) - 8 * x[1] * (x[2] - x[1]^2)
    g[2] = 4 * (x[2] - x[1]^2)
    H .= [1 + 8 * (3 * x[1]^2 - x[2]) -8x[1]; -8x[1] 4.0]
    return tl_f(x)
end
# An iterator's data that holds a trial linearization, for the default linearize!
struct TLData{T}
    tl::T
end
II.triallinearization(d::TLData) = d.tl

function tl_state(x0)
    ls = SOO.makehessiansystem(zeros(2, 2), :auto)
    data = SOO.OptimizationInternal(ls, x0)
    data.bestcost = SOO.computelinearization!(data, tl_fdf!, data.variables)
    return data, ls
end

@testset "TrialLinearization: without a spare (callbackchangesstate = true)" begin
    data, ls = tl_state([0.5, 0.5])
    tl = II.TrialLinearization(ls, True())
    @test tl.trial === nothing && SOO.trialsystem(tl) === nothing && !tl.ready
    II.settrial!(data, [0.1, 0.2])
    nc, ng = data.costs.count, data.gradients.count
    for likely in (false, true)
        @test II.trialcost!(tl, data, tl_f, tl_fdf!, likely) === tl_f([0.6, 0.7])
        @test !tl.ready
    end
    @test data.costs.count == nc + 2 && data.gradients.count == ng
    # The default linearize! linearizes at the current point
    g0 = copy(getb(ls))
    @test II.linearize!(TLData(tl), data, tl_fdf!) === tl_f([0.5, 0.5])
    @test data.linsystem === ls && getb(ls) == g0 && data.gradients.count == ng + 1
end

@testset "TrialLinearization: with a spare (callbackchangesstate = false)" begin
    data, ls = tl_state([0.5, 0.5])
    tl = II.TrialLinearization(ls, False())
    spare = tl.trial
    @test typeof(spare) === typeof(ls) && spare !== ls && SOO.trialsystem(tl) === spare && !tl.ready
    II.settrial!(data, [0.1, 0.2])
    nc, ng = data.costs.count, data.gradients.count
    # Not likely: f, and not ready
    @test II.trialcost!(tl, data, tl_f, tl_fdf!, false) === tl_f([0.6, 0.7])
    @test !tl.ready && data.costs.count == nc + 1 && data.gradients.count == ng
    # Likely: fdf! into the spare, which shares the step with the current system, and is ready
    g0 = copy(getb(ls))
    @test II.trialcost!(tl, data, tl_f, tl_fdf!, true) === tl_f([0.6, 0.7])
    @test tl.ready && tl.cost === tl_f([0.6, 0.7])
    @test data.costs.count == nc + 1 && data.gradients.count == ng + 1
    @test getb(ls) == g0                       # the current system is untouched
    @test getx(tl.trial) === getx(ls)          # sharex: one step storage
    gtrial = copy(getb(tl.trial))
    # discardtrial! makes it not ready: linearize! then linearizes at the current point
    II.discardtrial!(tl)
    @test !tl.ready
    @test II.linearize!(TLData(tl), data, tl_fdf!) === tl_f([0.5, 0.5])
    @test data.linsystem === ls && data.gradients.count == ng + 2
    # Ready: linearize! swaps the spare in, returns the stored cost, and evaluates nothing
    II.trialcost!(tl, data, tl_f, tl_fdf!, true)
    ng = data.gradients.count
    old = data.linsystem
    @test II.linearize!(TLData(tl), data, tl_fdf!) === tl_f([0.6, 0.7])
    @test data.gradients.count == ng && !tl.ready
    @test data.linsystem !== old && tl.trial === old
    @test getb(data.linsystem) == gtrial
    @test II.getstep(data) === getx(tl.trial)  # the step storage stays shared after the swap
    # reset! clears the flag and the stored cost
    II.trialcost!(tl, data, tl_f, tl_fdf!, true)
    II.reset!(tl)
    @test !tl.ready && isnan(tl.cost)
end

@testset "triallinearization: the default is nothing, and the default linearize! then linearizes" begin
    data, ls = tl_state([0.5, 0.5])
    @test II.triallinearization(:anything) === nothing
    @test SOO.trialsystem(:anything) === nothing
    ng = data.gradients.count
    @test II.linearize!(:anything, data, tl_fdf!) === tl_f([0.5, 0.5])
    @test data.gradients.count == ng + 1
end
