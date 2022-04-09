using Arpack
using Distributions
using LightGraphs, GraphIO
using LinearAlgebra, SparseArrays
using StatsBase


# 1. Symmetric Edge Flip
function privacy(; ϵ::Real = -1, p::Real = -1)
    if ϵ < 0
        return log((1 - p) / p)

    elseif p < 0
        return 1 / (1 + exp(ϵ))
    end
end

_flipSingleEdge(x, p) = rand(Bernoulli(p)) ? 1 - x : x

function edgeFlip(A; ϵ::Real = 1, p::Real = -1, parameters=nothing)
    if p < 0
        p = privacy(ϵ = ϵ)
    end

    n = size(A, 1)
    X = sparse([i>j ? _flipSingleEdge(A[i, j], p) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end;

function generate_rdpg(f, Z)
    n = size(Z, 1)

    if size(Z, 2) == 1
        X = sparse([i > j ? 1 * rand(Bernoulli(f(Z[i], Z[j]))) : 0 for i = 1:n, j = 1:n])
    else
        X = sparse([i > j ? 1 * rand(Bernoulli(min( f(Z[i, j]), 1))) : 0 for i = 1:n, j = 1:n])
    end

    return LightGraphs.LinAlg.symmetrize(X)
end;

function spectral_embeddings(A; d = 3, scale=true)
    λ, v = eigs(A, nev = d, maxiter=500)
    X = v * diagm(.√ abs.(λ))
    
    if scale
        X = StatsBase.standardize(ZScoreTransform, X, dims=1)
    end

    return X, λ, v
end;

function read_network(filename)
    return LightGraphs.LinAlg.symmetrize(adjacency_matrix(
        loadgraph(filename, "network", EdgeListFormat())
    ))
end;
