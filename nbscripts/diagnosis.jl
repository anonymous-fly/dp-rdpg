#Diagnosis
using DrWatson



begin
    include(srcdir("rdpg.jl"))
    import Main.rdpg
    using Distributions, LinearAlgebra, Plots, ProgressMeter, Random, Statistics, StatsBase
    using Pipe, Ripserer, UMAP
end



begin
    function subsample(X, a=1)
        sample(X |> rdpg.m2t, round(Int, size(X, 1)^a), replace=false)
    end

    function generate_graph(n; k=2, r=0.1, p=0.2)
        Z = rand(1:k, n)
        f = (x, y) -> r + p * (x == y)
        return rdpg.Adjacency(f, Z)
    end
end




function sim(n, d, e, order, p=2)
    ϵ = e

    A = generate_graph(n)
    X, _, _ = rdpg.spectralEmbed(A, d=d)
    DX = diagram(X |> subsample, dim_max=order)
    plt1 = @pipe X |> rdpg.m2t |> scatter(_, label="1", ratio=1, ma=0.1, msw=0)


    X_norm = scale_embeddings(X)
    DX_norm = diagram(X_norm |> subsample, dim_max=order)
    plt2 = @pipe X_norm |> rdpg.m2t |> scatter(_, label="1", ratio=1, ma=0.1, msw=0)


    A1 = rdpg.edgeFlip(A, ϵ=ϵ)
    X1, _ = rdpg.spectralEmbed(A1, d=d)
    X1_norm = scale_embeddings(X1)
    D1_norm = diagram(X1_norm |> subsample, dim_max=order)
    plt2 = @pipe X1 |> rdpg.m2t |> scatter(plt2, _, ratio=1, ma=0.5, msw=0)
    plt2 = @pipe X1_norm |> rdpg.m2t |> scatter(plt2, _, ratio=1, ma=0.1, msw=0)


    A2 = (A1 .- rdpg.τ(ϵ)^2) ./ rdpg.σ(ϵ)
    X2, _ = rdpg.spectralEmbed(A2, d=d)
    D2 = diagram(X2 |> subsample, dim_max=order)
    plt1 = @pipe X2 |> rdpg.m2t |> scatter(plt1, _, ratio=1, ma=0.5, msw=0)

    # println((:d_we, bottleneck_distance(DX, D2, p=p), :d_ne, bottleneck_distance(DX_norm, D1_norm, p=p)))

    return (bottleneck_distance(DX, D2, p=p), bottleneck_distance(DX_norm, D1_norm, p=p)), (plot(plt1, plt2), plot(plot(DX), plot(D2), title="we"), plot(plot(DX_norm), plot(D1_norm), title="ne"))
end

function sim_eps(eps; N, repeats=5, ribbon=true)
    μ = zeros(size(N, 1), 2)
    σ = zeros(size(N, 1), 2)
    @showprogress for (n, i) in zip(N, eachindex(N))
        # println("\n\n\nStarting with n=$n")
        Random.seed!(2022)
        tmp = [sim(n, 2, eps, 1)[1] for _ in 1:repeats]
        μ[i, :] = @pipe tmp |> rdpg.t2m |> mean(_, dims=1)
        σ[i, :] = @pipe tmp |> rdpg.t2m |> mean(0.25 .* _, dims=1)
    end

    plot(
        0:1e-10, 0:1e-10, la=0, ma=0,
        label="ϵ=$eps", xlabel="n", ylabel="Bottleneck distance",
        size=(400, 300)
    )
    plot!(N, μ[:, 1], ribbon=σ[:, 1], m=:o, label="ϵ known")
    plot!(N, μ[:, 2], ribbon=σ[:, 2], m=:o, label="ϵ unknown")
    savefig(plotsdir("eps-sims/e-$eps.pdf"))

    return results
end

N = [100, 200, 500, 1_000, 5_000] # , 2_000, 10_000
sim_eps(1; N=N)
sim_eps(3; N=N)
sim_eps(5; N=N)


using Plots
tmp = randn(100, 2)
plot(0:1e-10, 0:1e-10, la=0, ma=0, label="ϵ=$e", xlabel="n", ylabel="Bottleneck distance", size=(400, 400))
plot!(tmp[:, 1], m=:o, label="ϵ known")
plot!(tmp[:, 2], m=:o, label="ϵ unknown")