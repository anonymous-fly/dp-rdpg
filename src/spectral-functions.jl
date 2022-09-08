
function spectralEmbed(A; d=3, scale=false, restarts=200)
    λ, v = partialeigen(partialschur(A, nev=d, which=LM(), restarts=restarts)[1])
    X = v * diagm(.√abs.(λ))

    if scale
        X = StatsBase.standardize(ZScoreTransform, X, dims=1)
    end
    return X, λ, v
end


function oldspectralEmbed(A; d = 3, scale=false)
    λ, v = eigs(A, nev = d)
    X = v * diagm(.√ abs.(λ))
    
    if scale
        X = StatsBase.standardize(ZScoreTransform, X, dims=1)
    end
    return X, λ, v
end


function scale(x)
    return (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
end


function scale_embeddings(X)
    # c = cov(X)
    # U = eigvecs(c)
    # s = U * Diagonal(eigvals(c) .^ -0.5) * transpose(U)
    return (X .- mean(eachrow(X))') * (X'X ./ size(X,1) )^(-0.5)
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


