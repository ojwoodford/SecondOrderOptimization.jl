using Test

# rules.md 44: imports appear once, at the top of the module file, before its include lines; an included file of
# src/ has none. (The form of NLLSModel's test/imports.jl.) The extensions are modules of their own.
module ImportCheck

# The lines of the `using` and `import` statements in `ex`, not counting those of a submodule it defines
function importlines(ex, lines=Int[], line=0)
    ex isa Expr || return lines
    ex.head === :module && return lines
    if ex.head === :import || ex.head === :using
        push!(lines, line)
        return lines
    end
    for a in ex.args
        a isa LineNumberNode ? (line = a.line) : importlines(a, lines, line)
    end
    return lines
end
importlines(file::AbstractString) = importlines(Meta.parseall(read(file, String); filename=file))

# The body of the first module defined in `ex` (under its docstring, if it has one)
function modulebody(ex)
    ex isa Expr || return nothing
    ex.head === :module && return ex.args[3]
    for a in ex.args
        body = modulebody(a)
        body === nothing || return body
    end
    return nothing
end

# The module file's imports, and the line of its first top-level `include`
function moduleimports(file)
    body = modulebody(Meta.parseall(read(file, String); filename=file))
    line, first_include = 0, typemax(Int)
    for a in body.args
        a isa LineNumberNode && (line = a.line)
        if a isa Expr && a.head === :call && a.args[1] === :include
            first_include = line
            break
        end
    end
    return importlines(body), first_include
end

end # module ImportCheck

@testset "Imports once, in the module file (AS5)" begin
    root = dirname(@__DIR__)
    srcfiles = filter(endswith(".jl"), readdir(joinpath(root, "src"); join=true))
    mainfile = joinpath(root, "src", "SecondOrderOptimization.jl")
    lines, first_include = ImportCheck.moduleimports(mainfile)
    @test !isempty(lines) && maximum(lines) < first_include < typemax(Int)
    @testset "src/$(basename(file)) has no import" for file in filter(!=(mainfile), srcfiles)
        @test isempty(ImportCheck.importlines(file))
    end
    # The check sees an import in a file, but not in a submodule or a docstring
    mktempdir() do dir
        file = joinpath(dir, "m.jl")
        write(file, "f(x) = x\n\"\"\"\nusing A\n\"\"\"\ng(x) = x\nmodule Sub\nusing B\nend\nimport C: c\n")
        @test ImportCheck.importlines(file) == [9]
    end
end
