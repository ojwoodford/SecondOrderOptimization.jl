# Loads files of test/fixtures into Main, each once per session: a second `include` of a test file defines no new
# type and creates no new objective type, so in a warm session nothing is recompiled for the fixtures.
# A test file loads what it needs at its top: `include("fixtures/load.jl"); loadfixtures("tp3.jl")`.
if !isdefined(Main, :loadfixtures)
    const LOADED_FIXTURES = Set{String}()
    function loadfixtures(names::AbstractString...)
        for name in names
            path = joinpath(@__DIR__, name)
            path in LOADED_FIXTURES && continue
            Base.include(Main, path)
            push!(LOADED_FIXTURES, path)
        end
        return nothing
    end
end
