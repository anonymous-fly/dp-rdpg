function spectralEmbed(A; d=3, scale=false, restarts=200)
    # tmp = KrylovKit.eigsolve(A, d; krylovdim=2*d)
    # λ, v = tmp[1], hcat(tmp[2]...)
    λ, v = partialeigen(partialschur(A, nev=d, which=LM(), restarts=restarts, tol=1e-6)[1])
    X = v * diagm(.√abs.(λ))

    if scale
        X = StatsBase.standardize(ZScoreTransform, X, dims=1)
    end
    return X, λ, v
end

function scale(x)
    return (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
end


function scale_embeddings(X)
    return StatsBase.standardize(ZScoreTransform, X, dims=1)
end

function cluster_embeddings(X, d)
    iter = 25
    best = -1
    clusters = nothing

    for i = 1:iter
        res = kmeans(X', d)
        metric = mean(silhouettes(res, pairwise(Distances.Euclidean(), X, dims = 1)))

        if metric > best
            best = metric
            clusters = res.assignments
        end
    end

    return clusters
end