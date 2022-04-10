function Adjacency(f, Z)
    n = size(Z, 1)
    # X = sparse(zeros(n,n))

    if size(Z, 2) == 1
        X = sparse([i > j ? 1 * rand(Bernoulli(f(Z[i], Z[j]))) : 0 for i = 1:n, j = 1:n])
    else
        X = sparse([i > j ? 1 * rand(Bernoulli(min(f(Z[i, :], Z[j, :]), 1))) : 0 for i = 1:n, j = 1:n])
    end

    return LightGraphs.LinAlg.symmetrize(X)
end


function scale(x)
    return (x .- mean(x, dims=1)) ./ std(x, dims=1)
end


function generate_sbm(n, k, p, r)
    f = (x, y) -> r + p * (x == y)
    Z = rand(1:k, n)
    return Adjacency(f, Z)
end


function σ(ϵ)
    return √((exp(ϵ) - 1) / (exp(ϵ) + 1))
end

function τ(ϵ)
    return √(1 / (exp(ϵ) + 1))
end

function privacy(; ϵ::Real=-1, p::Real=-1)
    if ϵ < 0
        return log((1 - p) / p)

    elseif p < 0
        return 1 / (1 + exp(ϵ))
    end
end

_flipSingleEdge(x, p) = rand(Bernoulli(p)) ? 1 - x : x

function edgeFlip(A; ϵ::Real=1, p::Real=-1, parameters=nothing)
    if p < 0
        p = privacy(ϵ=ϵ)
    end

    n = size(A, 1)
    X = sparse([i > j ? _flipSingleEdge(A[i, j], p) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end


# 2. Laplace Edge Flip

_LapFlipEdge(x, ϵ) = x + rand(Laplace(0, 1 / ϵ)) > 0.5 ? 1 : 0

function laplaceFlip(A; ϵ::Real=1, parameters=nothing)

    n = size(A, 1)
    X = sparse([i > j ? _LapFlipEdge(A[i, j], ϵ) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end;