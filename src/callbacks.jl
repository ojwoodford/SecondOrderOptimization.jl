import Printf.@printf

nullcallback(cost, unusedargs...) = (cost, 0)

function printoutcallback(cost, x_trial, data, trailingargs...)
    prevcost = data.bestcost
    if data.iternum == 1
        prevcost = data.startcost
        println("iter      cost      cost_change    |step|")
        @printf("% 4d % 8e  % 4.3e   % 3.2e\n", 0, prevcost, 0, 0)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e\n", data.iternum, cost, prevcost - cost, fast_norm(getx(data.linsystem)))
    return cost, 0
end

function printoutcallback(cost, data, trradius::Float64)
    prevcost = data.bestcost
    if data.iternum == 1
        prevcost = data.startcost
        println("iter      cost      cost_change    |step|    tr_radius")
        @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", 0, prevcost, 0, 0, trradius)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", data.iternum, cost, prevcost - cost, fast_norm(getx(data.linsystem)), trradius)
    return cost, 0
end

struct CostTrajectory
    costs::Vector{Float64}
    times_ns::Vector{UInt64}
    trajectory::Vector{Vector{Float64}}

    function CostTrajectory()
        return new(sizehint!(Vector{Float64}(), 50), sizehint!(Vector{UInt64}(), 50), sizehint!(Vector{Vector{Float64}}(), 50))
    end
end

function Base.empty!(ct::CostTrajectory)
    empty!(ct.costs)
    empty!(ct.times_ns)
    empty!(ct.trajectory)
    return ct
end

function storecostscallback(costs::Vector{Float64}, cost, unusedargs...)
    push!(costs, cost)
    return (cost, 0)
end

function storecostscallback(store::CostTrajectory, cost, x_trial, data, unusedargs...)
    push!(store.costs, cost)
    push!(store.times_ns, Base.time_ns() - data.start.time_ns)
    push!(store.trajectory, Vector{Float64}(getx(data.linsystem)))
    return (cost, 0)
end

storecostscallback(store) = (args...) -> storecostscallback(store, args...)
