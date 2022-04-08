import Pkg;
Pkg.activate(".")

# include("../../code/src/rdpg.jl")
include("code/src/rdpg.jl")
import Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, ProgressBars
using LinearAlgebra, Distances, Manifolds, Manopt, Distributions, LaTeXStrings
using Plots, TSne, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase

begin
    N = [200, 500, 1000, 2000, 3000, 5000, 10000, 20000]
    reps = 5
    h = 0.5  # Bandwidth for Gaussian (trace-class) kernel
    M = Sphere(1) # Define the circle using the Manifolds package
    fun = (x, y) -> min(1, pdf(Normal(0, h), distance(M, x, y)))
end

function slice2d(x; f, dim = 1)
    n = length(x)
    m = length(x[1])
    return dim == 1 ? [f([x[i][j] for i in 1:n]) for j in 1:m] : [f([x[i][j] for j in 1:m]) for i in 1:n]
end

function one_iter(A; n, f, d = 3, reps = 1, type = "alpha", subsample = true)
    e = Any[]
    ne = Any[]
    K = length(f)

    if type == "alpha"
        dgm = X -> @pipe X |> rdpg._Matrix_to_ArrayOfTuples |> ripserer(Alpha(_), dim_max = 1)
    else
        dgm = X -> @pipe X |> rdpg._Matrix_to_ArrayOfTuples |> ripserer(_, sparse = true, dim_max = 1)
    end

    for i ∈ tqdm(1:reps)
        # idx = sample(1:size(A, 1), n, replace = false)
        # Av = A[idx, idx]
        Z = rdpg.randSphere(n, d = 1)
        Av = rdpg.Adjacency(fun, Z)

        Xn, _ = rdpg.spectralEmbed(Av, d = d, scale = false)
        Y = [@pipe Av |> rdpg.edgeFlip(_, ϵ = f[k](n)) |> rdpg.spectralEmbed(_, d = d, scale = false) |> _[1] for k in 1:K]
        X = [@pipe Av |> rdpg.edgeFlip(_, ϵ = f[k](n)) |> Ms(_, f[k](n)) |> rdpg.spectralEmbed(_, d = d, scale = false) |> _[1] for k in 1:K]

        if subsample
            id() = sample(1:n, maximum([floor(Int, (n) / (1.5 * log(n))), 200]), replace = false)
            Xn = Xn[id(), :]
            Y = [y[id(), :] for y in Y]
            X = [x[id(), :] for x in X]
        end

        Dns = @pipe Xn |> rdpg.scale |> dgm
        Dn = @pipe Xn |> dgm
        push!(ne, [Bottleneck()(Dns, @pipe Y[k] |> rdpg.scale |> dgm) for k in 1:K])
        push!(e, [Bottleneck()(Dn, @pipe X[k] |> dgm) for k in 1:K])
    end
    return (e, ne)
end


begin
    filename = "/storage/home/suv87/work/julia/grdpg/code/datasets/facebook_combined.txt"
    G = Graphs.loadgraph(filename, "graph_key", EdgeListFormat())
    A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize
end

Ms = (X, ϵ) -> (X .- rdpg.τ(ϵ)^2) ./ rdpg.σ(ϵ)^2
rate1 = n -> log(1 + (log(n) / √n))
rate2 = n -> log(1 + log(n) / (n^(1 / 30)))
rate3 = n -> log(1 + log(n)^2 / (n^(1 / 10)))

rate1 = n -> log(1 + ((log(n)^(1 / 2)) / (n^(1 / 8))))
rate2 = n -> log(1 + ((log(n)^(2)) / (n^(1 / 24))))
# plot(1:10000, rate1)
# plot!(1:10000, rate2)



rate = [rate1, rate2]
# rate = [x->0.5, x->1, x->2, x->5, x->10]

E = Any[]
NE = Any[]

for i in tqdm(1:length(N))
    res = one_iter(A, n = N[i], f = rate, d = 3, reps = 10, type = "rips", subsample = true)
    push!(E, res[1])
    push!(NE, res[2])
end

# eplt = plot(0, 0, markeralpha = 0, ylim = (0, 1))
# neplt = plot(0, 0, markeralpha = 0, ylim = (0, 1))
# k = 1
# for k in 1:length(rate)
#     y = E
#     z = NE
#     eplt = plot(
#         eplt, N,
#         slice2d(slice2d.(y, f = mean), f = x -> x[k], dim = 2),
#         ribbon = slice2d(slice2d.(y, f = std), f = x -> x[k], dim = 2), label = false
#     )
#     neplt = plot(
#         neplt, N,
#         slice2d(slice2d.(z, f = mean), f = x -> x[k], dim = 2),
#         ribbon = slice2d(slice2d.(z, f = std), f = x -> x[k], dim = 2), label = false
#     )
# end
# savefig(eplt, "./plots/eplt.pdf")
# savefig(neplt, "./plots/neplt.pdf")




eplt = plot(0, 0, markeralpha = 0, ylim = (0, 1.2), xlab=L"n", ylab=L"W_\infty", size=(500, 400))
k = 1;
eplt = plot(
    eplt, N, label = L"$\epsilon(n) = \omega \left( \log\{1 + ( \log n / n  )^{1/8}\}   \right)$", legend = :topright,
    slice2d(slice2d.(y, f = median), f = x -> x[k], dim = 2),
    ribbon = slice2d(slice2d.(y, f = mad), f = x -> x[k], dim = 2)
)
k = 2;
eplt = plot(
    eplt, N, label = L"$\epsilon(n) = \omega \left( \log\{1 + ( \log n / n  )^{1/24}\}   \right)$", legend = :topright,
    slice2d(slice2d.(y, f = median), f = x -> x[k], dim = 2),
    ribbon = slice2d(slice2d.(y, f = mad), f = x -> x[k], dim = 2)
)
title!("Privacy-adjusted embedding")
savefig(eplt, "./plots/eplt.pdf")



neplt = plot(0, 0, markeralpha = 0, ylim = (0, 1.2), xlab=L"n", ylab=L"W_\infty", size=(500, 400))
k = 1;
neplt = plot(
    neplt, N, label = L"$\epsilon(n) = \omega \left( \log\{1 + ( \log n / n  )^{1/8}\}   \right)$", legend = :topright,
    slice2d(slice2d.(z, f = median), f = x -> x[k], dim = 2),
    ribbon = slice2d(slice2d.(z, f = mad), f = x -> x[k], dim = 2)
)
k = 2;
neplt = plot(
    neplt, N, label = L"$\epsilon(n) = \omega \left( \log\{1 + ( \log n / n  )^{1/24}\}   \right)$", 
    slice2d(slice2d.(z, f = median), f = x -> x[k], dim = 2),
    ribbon = slice2d(slice2d.(z, f = mad), f = x -> x[k], dim = 2)
)
title!("Self-normalized embedding")
savefig(neplt, "./plots/neplt.pdf")
