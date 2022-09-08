function Adjacency(f, Z)
    n = size(Z, 1)
    if size(Z, 2) == 1
        X = sparse([i > j && rand() < f(Z[i], Z[j]) for i = 1:n, j = 1:n])
    else
        X = sparse([i > j && rand() < f(Z[i, :], Z[j, :]) for i = 1:n, j = 1:n])
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

_flipSingleEdge(x, p) = rand() < p ? 1 - x : x

function edgeFlip(A; ϵ::Real=1, p::Real=-1, parameters=nothing)
    if p < 0
        p = privacy(ϵ=ϵ)
    end
    return LightGraphs.LinAlg.symmetrize(_flipSingleEdge.(A, p))
end


# 2. Laplace Edge Flip

_LapFlipEdge(x, ϵ) = x + rand(Laplace(0, 1 / ϵ)) > 0.5 ? 1 : 0

function laplaceFlip(A; ϵ::Real=1, parameters=nothing)
    if p < 0
        p = privacy(ϵ=ϵ)
    end
    return LightGraphs.LinAlg.symmetrize(_LapFlipEdge.(A, p))
end;