using DrWatson
@quickactivate projectdir()

begin
    include(srcdir("rdpg.jl"))
    import Main.rdpg
    using PersistenceDiagrams, Pipe, Plots, ProgressMeter, Random, Ripserer, Statistics, StatsBase
    using Distributions, LinearAlgebra, UMAP
end



begin
    
    function scale_embeddings(X)
        return StatsBase.standardize(ZScoreTransform, X, dims=1)
    end

    function diagram(X, dim_max; alpha=false)
        points = tuple.(eachcol(X)...)
        if alpha
            dgm = ripserer(Alpha(points), dim_max=dim_max)
        else
            dgm = ripserer(points, dim_max=dim_max)
        end
        return dgm
    end

    function bottleneck_distances(X, Y, dim_max)
        DX = diagram(X, dim_max)
        DY = diagram(Y, dim_max)
        return [Bottleneck()(DX[d], DY[d]) for d in 1:dim_max+1]
    end

    function subsample(X, a=0.5)
        sample(X |> rdpg.m2t, round(Int, size(X, 1)^a), replace=false)
    end



    function generate_graph(n)
        σ = 1.0  # Bandwidth
        Z = (rdpg.randLemniscate(n) |> rdpg.t2m ).+  0.1 .* randn(n, 2)
        Z = Z |> rdpg.m2t
        # f = (x, y) -> min(1, pdf(Normal(0, σ), norm(x .- y)))
        f = (x, y) -> min(1, pdf(Laplace(0, σ), norm(x .- y)))
        return rdpg.Adjacency(f, Z)
    end

end


begin
    repeats = 5
    N = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
    ϵ = [1, 2, 5]
    n = length(N)
    max_dim=1
end


begin
    with_eps    = [zeros(repeats, n) for _ in 1:length(ϵ)];
    without_eps = [zeros(repeats, n) for _ in 1:length(ϵ)];
    prog = Progress(convert(Int, n * repeats * length(ϵ)))

    for i in 1:n

        A = generate_graph(N[i])
        X, _, _ = rdpg.spectralEmbed(A, d=d, scale=false)
        X_norm = scale_embeddings(X)
        DX = ripserer(X |> subsample, dim_max=max_dim)
        DX_norm = ripserer(X_norm |> subsample, dim_max=max_dim)

        for j in eachindex(ϵ), k in 1:repeats

            A1 = rdpg.edgeFlip(A, ϵ=ϵ[j])            
            X1, _ = rdpg.spectralEmbed(A1, d=d, scale=false)
            X1_norm = scale_embeddings(X1)
            D1_norm = ripserer(X1_norm |> subsample, dim_max=max_dim)
            without_eps[j][k, i] = Bottleneck()(DX_norm, D1_norm)

            A2 = (A1 .- rdpg.τ(ϵ[j])) ./ rdpg.σ(ϵ[j])^2
            X2, _ = rdpg.spectralEmbed(A2, d=d, scale=false)
            D2 = ripserer(X2 |> subsample, dim_max=max_dim)
            with_eps[j][k, i] = Bottleneck()(DX, D2)

            next!(prog)
        end
    end

end



begin
    plt1 = plot(title="ϵ publicly available")
    for k in eachindex(ϵ)
        plot!(plt1, N,
            with_eps[k] |> eachcol .|> skipmissing .|> median,
            # ribbon = with_eps[k] |> eachcol .|> skipmissing .|> std,
            label="ϵ=$(ϵ[k])",
            lw=3, fillapha=0.01
        )
    end
    plt1
end


begin
    plt2 = plot(title="ϵ not publicly available")
    for k in eachindex(ϵ)
        plot!(plt2, N,
            without_eps[k] |> eachcol .|> skipmissing .|> mean,
            #ribbon = without_eps[k] |> eachcol .|> skipmissing .|> std,
            label="ϵ=$(ϵ[k])",
            lw=3, fillapha=0.01,
        )
    end
    plt2    
end

lim = (0.01, 0.6)


plot(plot(plt1, ylim=lim), plot(plt2, ylim=lim), size=(600, 300))
