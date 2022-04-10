using DrWatson
@quickactivate "projectdir()"

includet(srcdir("rdpg.jl"))
import Main.rdpg
using PersistenceDiagrams, Pipe, Plots, ProgressMeter, Random, Ripserer, Statistics, StatsBase

begin

    # function scale_embeddings(X)
    #     return (X .- mean(eachrow(X))') * (X'X)^(-0.5)
    # end

    function diagram(X, dim_max; alpha=true)
        points = tuple.(eachcol(X)...)
        dgm = ripserer(Alpha(points), dim_max=dim_max)
        return dgm
    end

    function bottleneck_distances(X, Y, dim_max)
        DX = diagram(X, dim_max)
        DY = diagram(Y, dim_max)
        return [Bottleneck()(DX[d], DY[d]) for d in 1:0+1]
    end

    function generate_sbm(n, k, p, r)
        f = (x, y) -> r + p * (x == y)
        Z = rand(1:k, n)
        return rdpg.Adjacency(f, Z)
    end

    function simulate_one(A, d, epsilon, method)
        X, _, _ = rdpg.spectralEmbed(A, d=d + 1, scale=false)
        A_private = rdpg.edgeFlip(A, ϵ=epsilon)

        if method == :eps
            A_private = A_private .- rdpg.rdpg.privacy(ϵ=epsilon)
        end

        X_private, _, _ = rdpg.spectralEmbed(A_private, d=d + 1, scale=false)

        if method == :eps
            X_private = X_private ./ (1 - 2 * rdpg.rdpg.privacy(ϵ=epsilon))
        elseif method == :noeps
            X = scale_embeddings(X)
            X_private = scale_embeddings(X_private)
        end
        return bottleneck_distances(X, X_private, d)
    end



    function slice(V, fun=mean; dim=1, i=1)
        if dim == 1
            reshape(fun(V, dims=1), size(V, 3), size(V, 2), :)[i, :]
        elseif dim == 2
            reshape(fun(V, dims=1), size(V, 3), size(V, 2), :)[:, i]
        end
    end

    function scale_embeddings(X)
        c = cov(X)
        U = eigvecs(c)
        s = U * Diagonal(eigvals(c) .^ -0.5) * transpose(U)
        return X * s
    end

    function simulate_one(A, d, epsilon, method)
        X, _, _ = rdpg.spectralEmbed(A, d=d + 1, scale=false)

        A_private = rdpg.edgeFlip(A, ϵ=epsilon)

        if method == :eps
            A_private = A_private .- rdpg.privacy(ϵ=epsilon)
        end

        X_private, _, _ = rdpg.spectralEmbed(A_private, d=d + 1, scale=false)

        if method == :eps
            X_private = X_private ./ (1 - 2 * rdpg.privacy(ϵ=epsilon))
        elseif method == :noeps
            X = StatsBase.standardize(ZScoreTransform, X, dims=1)
            X_private = StatsBase.standardize(ZScoreTransform, X_private, dims=1)
        end

        return bottleneck_distances(X, X_private, 0)
    end

end;



begin
    p, r = 0.4, 0.1, 0.15
    clust = 3
    repeats = 10

    N = [50, 100, 200, 400]
    ϵ = [0.5, 1, 2, 4]
end


begin
    n = length(N)
    m = length(ϵ)

    ne1 = zeros(repeats, n, m)
    we1 = zeros(repeats, n, m)

    ne2 = zeros(repeats, n, m)
    we2 = zeros(repeats, n, m)

    ne3 = zeros(repeats, n, m)
    we3 = zeros(repeats, n, m)

    prog = Progress(convert(Int, n * m * repeats * 2))

    for i in 1:n
        for j in 1:m
            for k in 1:repeats
                A = generate_sbm(N[i], clust, p, r)
                for method in [:eps, :noeps]
                    results = simulate_one(A, 2, ϵ[j], method)
                    # fields = [n, ϵ, method]
                    # append!(fields, results)
                    # println(join(fields, "\t"))
                    if method == :eps
                        we1[k, i, j] = results # [1]
                        we2[k, i, j] = results # [2]
                        we3[k, i, j] = results # [3]
                    else # 
                        ne1[k, i, j] = results # [1]
                        ne2[k, i, j] = results # [2]
                        ne3[k, i, j] = results # [3]
                    end

                    next!(prog)
                end
            end
        end
    end
end



begin
    plt1 = plot(title="ϵ publicly available")
    i = 1
    plot!(plt1, N, slice(we1, mean; dim=1, i=i), ribbon=slice(we1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
    i = 2
    plot!(plt1, N, slice(we1, mean; dim=1, i=i), ribbon=slice(we1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
    i = 3
    plot!(plt1, N, slice(we1, mean; dim=1, i=i), ribbon=slice(we1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
    # i=4; plot!(plt1, N, slice(we1, mean; dim=1, i=i), ribbon=slice(we1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
end


begin
    plt2 = plot(title="ϵ not publicly available")
    i = 1
    plot!(plt2, N, slice(ne1, mean; dim=1, i=i), ribbon=slice(ne1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
    i = 2
    plot!(plt2, N, slice(ne1, mean; dim=1, i=i), ribbon=slice(ne1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
    i = 3
    plot!(plt2, N, slice(ne1, mean; dim=1, i=i), ribbon=slice(ne1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
    # i=4; plot!(plt2, N, slice(ne1, mean; dim=1, i=i), ribbon=slice(ne1, std; dim=1, i=i), label="ϵ = $(ϵ[i])")
end