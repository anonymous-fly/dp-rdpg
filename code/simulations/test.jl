# Makes sure we're using the same version of the packages. 
import Pkg; Pkg.activate(pwd()); Pkg.instantiate();

using Plots, Random, Distributions, Pipe, LinearAlgebra
using Ripserer, PersistenceDiagramsBase

# A function which generates n points from a circle with some radial noise.
function randCircle(;n = 100, σ = 0.05)
    return @pipe rand(Uniform(-π, π), n) .|> [cos(_), sin(_)] |> _ .* rand(Normal(1, σ), n) |> Tuple.(eachcol(_)...)
end


# Generate sample points
Random.seed!(2021)
X = randCircle(; n = 500, σ = 0.2)
scatter(X, ratio = 1, label = "", c = :firebrick)

# Compute Persistence Diagrams
@time D1 = ripserer(X)
@time D2 = ripserer(EdgeCollapsedRips(X))
@time D_alpha = ripserer(Alpha(X))

D3 = Array{PersistenceDiagram, 1}(undef, 2)
for dim ∈ 1:2
    D3[dim] = PersistenceDiagram(D_alpha[dim].intervals,
                (; D_alpha[dim].meta..., threshold = D2[1].meta.threshold) )
end

plot(
    plot(D1, title = "Vanilla Rips"),
    plot(D2, title = "EdgeCollapsed Rips"),
    plot(D3, title = "Alpha Complex"),
    layout = (1, 3)
)
