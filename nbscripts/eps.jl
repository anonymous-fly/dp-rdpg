using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
import Main.rdpg
using PersistenceDiagrams, Pipe, Plots, ProgressMeter, Random, Ripserer, Statistics, StatsBase
using Distributions, LinearAlgebra, UMAP
using ProgressMeter
using Random


function scale_embeddings(X)
    return StatsBase.standardize(ZScoreTransform, X, dims=1)
end

function diagram(X; dim_max)
    ripserer(X |> Alpha, dim_max=dim_max)
end

function bottleneck_distances(X, Y, dim_max)
    DX = diagram(X, dim_max)
    DY = diagram(Y, dim_max)
    return [Bottleneck()(DX[d], DY[d]) for d in 1:dim_max+1]
end

function bottleneck_distance(Dx, Dy; order=nothing, p=Inf)
    order = isnothing(order) ? 0 : order
    dx, dy = Dx[1+order], Dy[1+order]
    m = max(0, min(length.((dx, dy))...) .- 1)
    dx = dx[end-m:end-1]
    dy = dy[end-m:end-1]
    return norm(map((x, y) -> (x .- y) .|> abs |> maximum, dx, dy), p)
end

function subsample(X, a=1)
    sample(X |> rdpg.m2t, round(Int, size(X, 1)^a), replace=false)
end

function generate_graph(n; k=3, r=0.1, p=0.7)
    Z = rand(1:k, n)
    f = (x, y) -> r + p * (x == y)
    return rdpg.Adjacency(f, Z)
end

repeats = 50
N = [100, 500, 1000, 1500, 2000, 2500]
ϵ = [0.5, 1, 2]
n = length(N)
max_dim = 1
d = 2
order = 1;


begin
    with_eps = [zeros(repeats, n) for _ in 1:length(ϵ)]
    without_eps = [zeros(repeats, n) for _ in 1:length(ϵ)]
    prog = Progress(convert(Int, n * repeats * length(ϵ)))

    for i in 1:n
        Random.seed!(2022)
        A = generate_graph(N[i])

        X, _, _ = rdpg.spectralEmbed(A, d=d)
        DX = diagram(X |> subsample, dim_max=order)

        X_norm = scale_embeddings(X)
        DX_norm = diagram(X_norm |> subsample, dim_max=order)


        for j in eachindex(ϵ), k in 1:repeats

            A1 = rdpg.edgeFlip(A, ϵ=ϵ[j])
            X1, _ = rdpg.spectralEmbed(A1, d=d)
            X1_norm = scale_embeddings(X1)
            D1_norm = diagram(X1_norm |> subsample, dim_max=order)
            without_eps[j][k, i] = bottleneck_distance(DX_norm, D1_norm, order=order, p=1)


            A2 = (A1 .- rdpg.τ(ϵ[j])) ./ rdpg.σ(ϵ[j])^2
            X2, _ = rdpg.spectralEmbed(A2, d=d)
            D2 = diagram(X2 |> subsample, dim_max=order)
            with_eps[j][k, i] = bottleneck_distance(DX, D2, order=order, p=1)

            next!(prog)
        end
    end
end