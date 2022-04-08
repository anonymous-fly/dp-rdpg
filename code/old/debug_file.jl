using LinearAlgebra, SparseArrays, Arpack
using JuMP, Gurobi
using LightGraphs, Clustering, Distances
using Random, StatsBase, Statistics, Distributions
using GraphPlot, Plots, LaTeXStrings, ProgressBars
using HomotopyContinuation, Suppressor
using Manifolds, Manopt, Ripserer, PersistenceDiagrams
import Manifolds: distance

#= 
RCall is used to call `tdaunif::sample_lemniscate_gerono` in R, 
which samples points uniformly from a Lemniscate. Make sure you run:
        $ install.packages("tdaunif")
in your R session before running the code for the Lemniscate in Julia.
 =#
using RCall

include("mechanisms.jl")
include("spectral-functions.jl")


## Johnson-Lindenstrauss
function JLMechanism(A; ϵ::Real = 1, δ::Real = 0.001, η::Real = 0.05, ν::Real = 0.05)
    r = ceil(8 * log(2 / ν) / η^2)
    w = sqrt(32 * r * log(2 / δ)) * log(4*r / δ) / ϵ
    
    n = size(A, 1)
    M = randn((Int(r), binomial(n, 2)))

    # incidence matrix after transforming adjacency matrix
    A = Array(A) .* (1 - w/n) .+ w/n
    B = full_incidence_matrix(A)

    L = B' * M' * M * B ./ r

    A = sparse(replace(x -> x < -0.5 ? 1 : 0, L))
    A[diagind(A)] .= 0

    return A
end

function Evaluate(A, Z; M=edgeFlip, d=3, sbm=true, plt=false, log_trans=true, kwargs...)

    B = M(A; kwargs...);
    Xᵇ, _ = SpectralEmbed(A)
    X, λ, v = SpectralEmbed(B);

    results = Dict()

    push!(results, "density" => relativeDensity(A, B));
    push!(results, "adjerror" => sum(abs.(B - A)) / 2);
    
    push!(results, 
    "bottleneck" => bottleneck_distance(Xᵇ, X, log_trans=log_trans, plt_scatter=plt, plt_diagram=plt))

    if sbm
        Clust = cluster_embeddings(X, d);
        Accuracy = clustering_accuracy(Clust, Z);
        push!(results, "accuracy" => Accuracy)
    end

    println("\n\n################################ \n")
    println("""
    Summary: \n \n 
    Mechanism = $M
    $(tuple.(kwargs...))
    |A| = $(sum(A)) 
    sbm = $sbm \n""")

    for (key, value) in results
        println("$key = $value")
    end

    return results
end


function simulation(f, Z;
    M = edgeFlip, 
    ϵ = Inf, 
    runs=10, 
    parameters=nothing, 
    d=3,
    metrics = ["density", "bottleneck", "adjerror"],
    kwargs...
    )

    μ = zeros(length(ϵ),length(metrics));
    σ = zeros(length(ϵ),length(metrics));
    results = []

    for i = tqdm(1:runs)
        A = Adjacency(f, Z);
        push!(results , []); results[i] = []
        for j = tqdm(1:length(ϵ))
            push!(results[i], Evaluate(A, Z; M=M, ϵ=ϵ[j], kwargs...));
        end
    end

    for i = 1:length(metrics)
        for j in 1:length(ϵ)
            μ[j, i] = mean([results[u][j][metrics[i]] for u in 1:runs])
            σ[j, i] = std([results[u][j][metrics[i]] for u in 1:runs])
        end
    end

    return μ, σ, results

end

ϵ = 1
n = 100
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = Adjacency(f, Z);

runs = 1
metrics = ["density", "bottleneck", "adjerror"]

μ_lflip, σ_lflip = simulation(f, Z; M=JLMechanism, ϵ=ϵ, runs=runs, metrics=metrics, δ=1e-5);
