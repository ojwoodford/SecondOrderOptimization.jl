using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Static, Test
using DampedLinearSolve: getA, getb, getx, dampen!, solve!, Success, max_diag, getdiag!, quadratic_model_change, quadform,
    isfinitesystem, systemeltype, dampingfloor
const SOO = SecondOrderOptimization
const II = SecondOrderOptimization.IteratorInterface
const update = SOO.update   # OptimizationVariableInterface's

# The iterator interface: each function against the linear-system call it wraps, on a static, a dense and a sparse
# system of both objectives

# f = ½‖r‖², r = (x₁ − 1, 2(x₂ − x₁²), x₁ + x₂ − 3): three residuals in two variables
ii_r(x) = (x[1] - 1, 2 * (x[2] - x[1]^2), x[1] + x[2] - 3)
ii_J(x) = (1.0, -4 * x[1], 1.0, 0.0, 2.0, 1.0)   # column-major, 3 × 2
ii_f(x) = 0.5 * sum(abs2, ii_r(x))
function ii_resjac!(r, J, x)
    rv, Jv = ii_r(x), ii_J(x)
    for i in 1:3
        r[i] = rv[i]
        J[i, 1] = Jv[i]
        J[i, 2] = Jv[3 + i]
    end
    return ii_f(x)
end
function ii_fdf!(g, H, x)
    rv, Jv = ii_r(x), ii_J(x)
    J = [Jv[1] Jv[4]; Jv[2] Jv[5]; Jv[3] Jv[6]]
    g .= J' * collect(rv)
    M = J' * J
    if H isa SparseMatrixCSC
        for j in 1:2, k in nzrange(H, j)
            nonzeros(H)[k] = M[rowvals(H)[k], j]
        end
    else
        H .= M
    end
    return ii_f(x)
end

const II_CASES = (
    ("static Hessian", () -> SVector(0.5, 0.5), HessianObjective(ii_f, ii_fdf!)),
    ("dense Hessian", () -> [0.5, 0.5], HessianObjective(ii_f, ii_fdf!; H = zeros(2, 2))),
    ("sparse Hessian", () -> [0.5, 0.5], HessianObjective(ii_f, ii_fdf!; H = sparse(ones(2, 2)))),
    ("static Jacobian", () -> SVector(0.5, 0.5), JacobianObjective(ii_f, ii_resjac!, 3)),
    ("dense Jacobian", () -> [0.5, 0.5], JacobianObjective(ii_f, ii_resjac!, zeros(3, 2))),
    ("sparse Jacobian", () -> [0.5, 0.5], JacobianObjective(ii_f, ii_resjac!, sparse(ones(3, 2)))),
)

# The loop's state on a system linearized at x0, and a second system linearized identically, to call the contract on
function ii_state(x0, obj)
    ls = SOO.objective_linearsystem(x0, obj, :auto)
    data = SOO.OptimizationInternal(ls, x0)
    fdf! = SOO.linearizer(obj)
    data.bestcost = SOO.computelinearization!(data, fdf!, data.variables)
    ref = SOO.objective_linearsystem(x0, obj, :auto)
    refdata = SOO.OptimizationInternal(ref, x0)
    SOO.computelinearization!(refdata, fdf!, refdata.variables)
    return data, ls, ref, fdf!
end

@testset "IteratorInterface: $name" for (name, mkx, obj) in II_CASES
    options = OptimizationOptions()
    data, ls, ref, fdf! = ii_state(mkx(), obj)
    v = [0.3, -0.7]
    @testset "the read-only functions" begin
        @test II.getgradient(data) === getb(ls)
        @test II.getstep(data) === getx(ls)
        s = II.similarstep(ls)
        @test typeof(s) === typeof(getx(ls)) && size(s) == size(getx(ls)) && s !== getx(ls)
        @test II.maxdiagonal(data) === max_diag(ref)
        out = zeros(2)
        @test II.getdiagonal!(out, data) === out
        @test out == getdiag!(zeros(2), ref)
        @test II.predictedchange(data, v) === quadratic_model_change(ref, v)
        @test II.curvature(data, v) === quadform(getA(ref), v)
        @test II.systemisfinite(data) === isfinitesystem(ref) === true
        @test II.systemeltype(data) === systemeltype(ref) === Float64
        @test II.dampingfloor(data) === dampingfloor(ref)
        @test II.getbestcost(data) === data.bestcost === ii_f(mkx())
    end
    @testset "solvestep! solves as damped, negates, and counts one solve per call" begin
        for (args, damp!) in (((), ls -> nothing), ((0.5,), ls -> dampen!(ls, 0.5)), ((0.5, [2.0, 3.0]), ls -> dampen!(ls, 0.5, [2.0, 3.0])))
            n = data.solves.count
            @test II.solvestep!(data, options, args...) === true
            @test data.solves.count == n + 1
            damp!(ref)
            @test solve!(ref, options.linsolver) == Success
            @test II.getstep(data) == -getx(ref)
        end
        # The damping stays set until it is changed; undamp! clears it
        II.undamp!(data)
        dampen!(ref, 0.0)
        @test II.solvestep!(data, options) && solve!(ref, options.linsolver) == Success
        @test II.getstep(data) == -getx(ref)
        @test II.predictedchange(data, II.getstep(data)) === quadratic_model_change(ref, getx(ref) .* -1)
    end
    @testset "settrial!, trialcost! and linearizetrial!" begin
        x0 = mkx()
        II.solvestep!(data, options)
        @test II.settrial!(data) === nothing
        @test data.varnext == update(x0, II.getstep(data))
        @test II.settrial!(data, v) === nothing
        @test data.varnext == update(x0, v)
        nc, ng = data.costs.count, data.gradients.count
        @test II.trialcost!(data, ii_f) === ii_f(update(x0, v))
        @test data.costs.count == nc + 1 && data.gradients.count == ng
        @test II.linearizetrial!(data, fdf!) === ii_f(update(x0, v))
        @test data.costs.count == nc + 1 && data.gradients.count == ng + 1
        # The current system now holds the linearization at the trial point
        at = SOO.objective_linearsystem(x0, obj, :auto)
        atdata = SOO.OptimizationInternal(at, x0)
        SOO.computelinearization!(atdata, fdf!, update(x0, v))
        @test II.getgradient(data) == getb(at)
        @test II.maxdiagonal(data) === max_diag(at)
        @test data.variables == x0   # the current point is the loop's to move
    end
end

@testset "IteratorInterface: a failed solve returns false" begin
    zf(x) = 0.0
    zfdf!(g, H, x) = (g .= 1.0; 0.0)   # H stays zero
    for (x0, H) in ((SVector(0.0, 0.0), nothing), ([0.0, 0.0], zeros(2, 2)))
        data, ls, ref, fdf! = ii_state(x0, HessianObjective(zf, zfdf!; H))
        n = data.solves.count
        @test II.solvestep!(data, OptimizationOptions(linsolver = :cholesky)) === false
        @test data.solves.count == n + 1
    end
    # A non-finite matrix
    nf(x) = 0.0
    nfdf!(g, H, x) = (g .= 1.0; H[1, 1] = NaN; H[2, 2] = 1.0; 0.0)
    data, = ii_state([0.0, 0.0], HessianObjective(nf, nfdf!; H = zeros(2, 2)))
    @test II.systemisfinite(data) === false
    @test II.LINSOLVER_FAILED === SOO.LINSOLVER_FAILED && isnan(II.LINSOLVER_FAILED)
end

# The submodule's API, pinned (rules.md 10): its exported names, each with a docstring; and its functions are the
# parent's (systemeltype and dampingfloor DampedLinearSolve's), so that a method added through it extends them
const ITERATOR_INTERFACE_API = sort([
    :IteratorInterface,
    :LINSOLVER_FAILED,
    :construct,
    :curvature,
    :dampingfloor,
    :getbestcost,
    :getdiagonal!,
    :getgradient,
    :getstep,
    :iterate!,
    :iteratorname,
    :linearize!,
    :linearizetrial!,
    :maxdiagonal,
    :predictedchange,
    :reset!,
    :restrictedstep,
    :settrial!,
    :similarstep,
    :solvestep!,
    :systemeltype,
    :systemisfinite,
    :trialcost!,
    :undamp!,
])
@testset "IteratorInterface: the API" begin
    @test issorted(ITERATOR_INTERFACE_API)
    @test sort(names(II)) == ITERATOR_INTERFACE_API
    docs = merge(Base.Docs.meta(SOO), Base.Docs.meta(II))
    @testset "$name has a docstring" for name in ITERATOR_INTERFACE_API
        @test haskey(docs, Base.Docs.Binding(II, name))
    end
    for name in ITERATOR_INTERFACE_API
        name === :IteratorInterface && continue
        owner = name in (:systemeltype, :dampingfloor) ? SOO.DampedLinearSolve : SOO
        @test getfield(II, name) === getfield(owner, name)
    end
    @test :IteratorInterface in names(SOO)
end
