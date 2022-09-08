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
        m = max(0, min(length.((dx, dy))...) .- 2)
        dx = dx[end-m:end]
        dy = dy[end-m:end]
        return norm(map((x, y) -> (x .- y) .|> abs |> maximum, dx, dy), p)
    end

    function subsample(X, a=0.85)
        sample(X |> rdpg.m2t, round(Int, size(X, 1)^a), replace=false)
    end

    function generate_graph(n)
        σ = 0.75  # Bandwidth
        Z = (rdpg.randLemniscate(n) |> rdpg.t2m) .+ 0.1 .* randn(n, 2)
        Z = Z |> rdpg.m2t
        f = (x, y) -> pdf(Laplace(0, σ), norm(x .- y))
        return rdpg.Adjacency(f, Z)
    end
end


begin
    repeats = 20
    N = [100, 500, 1000, 2000, 5000, 10_000]
    ϵ = [0.5, 1, 2]
    n = length(N)
    max_dim = 1
    d = 2
    order = 0
end

with_eps = [zeros(repeats, n) for _ in 1:length(ϵ)];
without_eps = [zeros(repeats, n) for _ in 1:length(ϵ)];
prog = ProgressMeter.Progress(convert(Int, n * repeats * length(ϵ)))

for i in 1:n
    
    Random.seed!(2022)
    A = generate_graph(N[i])
    
    X, _, _ = rdpg.spectralEmbed(A, d=d)
    DX = diagram(X |> subsample, dim_max=order)
    
    X_norm = scale_embeddings(X)
    DX_norm = diagram(X_norm |> subsample, dim_max=order)
    

    for j in eachindex(ϵ), k in 1:repeats
        println("i=$i, j=$j, k=$k")
        A1 = rdpg.edgeFlip(A, ϵ=ϵ[j])
        X1, _ = rdpg.spectralEmbed(A1, d=d)
        X1_norm = scale_embeddings(X1)
        D1_norm = diagram(X1_norm |> subsample, dim_max=order)
        without_eps[j][k, i] = bottleneck_distance(DX_norm, D1_norm, order=order, p=2)


        A2 = (A1 .- rdpg.τ(ϵ[j])) ./ rdpg.σ(ϵ[j])^2
        X2, _ = rdpg.spectralEmbed(A2, d=d)
        D2 = diagram(X2 |> subsample, dim_max=order)
        with_eps[j][k, i] = bottleneck_distance(DX, D2, order=order, p=2)

        # next!(prog)
    end
end



begin
    plt1 = plot(title="ϵ publicly available")
    for k in eachindex(ϵ)
        plot!(plt1, N,
            with_eps[k] |> eachcol .|> mean,
            ribbon = 0.5 .* with_eps[k] |> eachcol .|> std,
            label="k=$(ϵ[k])",
            lw=3, fillapha=0.01,
        )
    end
    plt1
end

savefig(plt1, plotsdir("eps-sims/test1.pdf"))

begin
    plt2 = plot(title="ϵ not publicly available")
    for k in eachindex(ϵ)
        plot!(plt2, N,
            without_eps[k] |> eachcol .|> median,
            ribbon=0.5 .* without_eps[k] |> eachcol .|> mad,
            label="ϵ=$(ϵ[k])",
            lw=3, fillapha=0.01)
    end
    plt2
end

savefig(plt2, plotsdir("eps-sims/test2.pdf"))
