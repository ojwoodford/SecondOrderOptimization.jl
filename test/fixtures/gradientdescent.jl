# Gradient descent, the user-defined iterator of the tutorial (docs/src/tutorials/custom_iterator.md), registered as
# :gradientdescent. The code is not copied here: the tutorial's first `julia` code block, which defines the iterator, is
# read from the page and evaluated, so that the tutorial and this fixture are one source.
let page = joinpath(@__DIR__, "..", "..", "docs", "src", "tutorials", "custom_iterator.md")
    blocks = [m[1] for m in eachmatch(r"```julia\n(.*?)```"s, read(page, String))]
    code = first(blocks)
    occursin("function IteratorInterface.iterate!(gdd::GradientDescentData", code) || error("The first julia block of $page does not define the gradient-descent iterator")
    include_string(Main, code, page)
end
