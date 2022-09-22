using Ripserer: birth, death, ripserer, PersistenceInterval, PersistenceDiagram

function bottleneck_distance(Dx, Dy; order=nothing, p=Inf)
    order = isnothing(order) ? 0 : order
    dx, dy = Dx[1+order], Dy[1+order]
    m = max(0, min(length.((dx, dy))...) .- 2)
    dx = dx[end-m:end]
    dy = dy[end-m:end]
    return norm(map((x, y) -> (x .- y) .|> abs |> maximum, dx, dy), p)
end


_log_transform_interval(x, minval) = PersistenceInterval( tuple(max.(log.(x), fill(minval, 2))...) )

function log_transform_diagram(D)
    logD = deepcopy(D)
    minval = log(maximum(D[1][1]))
    for i in eachindex(logD)
        logD[i] = PersistenceDiagram(_log_transform_interval.(logD[i], minval), dim=i-1)
    end
    return logD
end

function diagram(X::Matrix{T}; d=2, log_trans=false, alpha=true) where {T}
    points = tuple.(eachcol(X)...)
    if alpha
        dgm = ripserer(Alpha(points), dim_max=d)
    else
        dgm = ripserer(EdgeCollapsedRips(points), dim_max=d)
    end
    return log_trans ? log_transform_diagram(dgm) : dgm
end

function diagram(X; dim_max=0)
    dgm = ripserer(X |> Alpha, dim_max=dim_max)
    return [replace(x -> death(x) == Inf ? PersistenceInterval(birth(x), threshold(d)) : x, d) for d in dgm]
end

function randLemniscate(n; s = 0)
    θ = range(0, 2π, n)
    signal = [sin.(θ) cos.(θ) .* sin.(θ)]
    noise = s .* randn(size(signal))
    return tuple.(eachcol(signal .+ noise)...)
end

function randCircle(n; s=0)
    θ = range(0, 2π, n)
    signal = [cos.(θ) sin.(θ)]
    noise = s .* randn(size(signal))
    return tuple.(eachcol(signal .+ noise)...)
end

function randSphere(n; d = 2)
    X = randn(n, d+1)
    return [tuple(x...) ./ norm(x) for x in eachrow(X)]
end