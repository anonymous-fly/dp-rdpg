ENV["GKSwstype"] = "100"

import Pkg;
Pkg.activate(pwd());
Pkg.instantiate();

using Ripserer, PersistenceDiagrams, Plots, ProgressBars, LaTeXStrings, ProgressMeter
include("../code/networks.jl")

function generate_sbm(n, k, p, r)
    f = (x, y) -> r + p * (x == y)
    Z = rand(1:k, n)
    return generate_rdpg(f, Z)
end

function diagram(X, dim_max; alpha=true)
    points = tuple.(eachcol(X)...)
    dgm = ripserer(Alpha(points), dim_max=dim_max)
    return dgm
end

function bottleneck_distances(X, Y, dim_max)
    DX = diagram(X, dim_max)
    DY = diagram(Y, dim_max)
    return [Bottleneck()(DX[d], DY[d]) for d in 1:dim_max+1]
end

function scale_embeddings(X)
    # c = cov(X)
    # U = eigvecs(c)
    # s = U * Diagonal(eigvals(c) .^ -0.5) * transpose(U)
    return (X .- mean(eachrow(X))') * (X'X)^(-0.5)
end

function simulate_one(A, d, epsilon, method)
    # Note: we add one to d, so don't add one yourself!
    X, _, _ = spectral_embeddings(A, d=d + 1, scale=false)

    # A_private = edge_flip(A, ϵ = epsilon)
    A_private = edgeFlip(A, ϵ=epsilon)

    if method == :eps
        A_private = A_private .- privacy(ϵ=epsilon)
    end

    X_private, _, _ = spectral_embeddings(A_private, d=d + 1, scale=false)

    if method == :eps
        X_private = X_private ./ (1 - 2 * privacy(ϵ=epsilon))
    elseif method == :noeps
        X = scale_embeddings(X)
        X_private = scale_embeddings(X_private)
        # X = StatsBase.standardize(ZScoreTransform, X, dims=1)
        # X_private = StatsBase.standardize(ZScoreTransform, X_private, dims=1)
    end

    # return maximum(bottleneck_distances(X, X_private, d+1))
    # why not record all dimensions and combine later?
    return bottleneck_distances(X, X_private, d)
end


function V(V, fun=mean; slice=1, i=1)
    if slice == 1
        reshape(fun(V, dims=1), size(V, 3), size(V, 2), :)[i, :]
    elseif slice == 2
        reshape(fun(V, dims=1), size(V, 3), size(V, 2), :)[:, i]
    end
end

function generate_sbm(n, k, p, r)
    f = (x, y) -> r + p * (x == y)
    Z = rand(1:k, n)
    return generate_rdpg(f, Z)
end

p, r = 0.4, 0.1, 0.15
clust = 3
repeats = 10
N = [50, 150, 300, 450, 600, 900]

n = length(N)

ne = zeros(repeats, n)
we1 = zeros(repeats, n)
we2 = zeros(repeats, n)
we3 = zeros(repeats, n)

prog = Progress(convert(Int, n * repeats))

for i in 1:n
    for k in 1:repeats

        A = generate_sbm(N[i], clust, p, r)

        for method in [:eps]

            # ϵ = log(1 + ((log(N[i]))/N[i])^(1/72))
            ϵ1 = log(N[i])^(1 / 3)
            ϵ2 = log(N[i])^(2 / 3)
            ϵ3 = log(N[i])^(1 / 1)

            results1 = simulate_one(A, 2, ϵ1, method)
            results2 = simulate_one(A, 2, ϵ2, method)
            results3 = simulate_one(A, 2, ϵ3, method)
            # println("n=$(N[i]), k=$k, ϵ = $ϵ, method=$method, result=$(results[1])")

            if method == :eps
                we1[k, i] = results1[1]
                we2[k, i] = results2[1]
                we3[k, i] = results3[1]
            else
                ne[k, i] = results[1]
            end

        end
        next!(prog)
    end
end


plt = plot(title="ϵ=logᵏ(n)")
plot!(plt, N, mean(we1, dims=1)', ribbon=std(we1, dims=1), label="k=0.33", linewidth=2, fillapha=0.01)#, markershape=:o)
plot!(plt, N, mean(we2, dims=1)', ribbon=std(we2, dims=1), label="k=0.66", linewidth=2, fillapha=0.01)#, markershape=:o)
plot!(plt, N, mean(we3, dims=1)', ribbon=std(we3, dims=1), label="k=1.00", linewidth=2, fillapha=0.01)#, markershape=:o)
xlabel!("n")
ylabel!("Bottleneck distance")

savefig(plt, "./plots/convergence.svg")
savefig(plt, "./plots/convergence.pdf")

