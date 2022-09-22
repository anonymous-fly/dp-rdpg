begin
    using DrWatson
    include(srcdir("rdpg.jl"))
    using Main.rdpg
    using LinearAlgebra, Plots, ProgressMeter, Random, Pipe
    using PersistenceDiagrams, Ripserer, Statistics, StatsBase
end


begin
    function generate_sbm_sparse(n, k, p, r)
        # f = (x, y) -> (r + p * (x == y)) * (log(n) / n^(0.1))
        f = (x, y) -> (r + p * (x == y)) * (log(n) / n^(1/3))
        # f = (x, y) -> (r + p * (x == y)) * log(n)^4 / n^(1/2)
        Z = rand(1:k, n)
        return rdpg.Adjacency(f, Z)
    end

    function generate_data(n, ϵ, params, method=:dense)
        if method == :dense
            A = generate_sbm_dense(n, params.clust, params.p, params.r)
        else
            A = generate_sbm_sparse(n, params.clust, params.p, params.r)
        end

        X, _, _ = rdpg.spectralEmbed(A, d=params.d, restarts=1000)
        # Add small perturbation to avoid degenerate simplices in Alpha complex
        X = X .+ randn(size(X)...) .* 1e-10
        Dx = rdpg.diagram(X |> rdpg.subsample, dim_max=params.order)

        B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)^2) ./ rdpg.σ(ϵ)^2
        Y, _ = rdpg.spectralEmbed(B, d=params.d, restarts=1000)

        # Add small perturbation to avoid degenerate simplices in Alpha complex
        Y = Y .+ randn(size(Y)...) .* 1e-10
        Dy = rdpg.diagram(Y |> rdpg.subsample, dim_max=params.order)

        return rdpg.bottleneck_distance(Dx, Dy, order=params.order, p=params.q)
    end


    function one_sim(f, N, params, repeats=5, method=:dense)
        n = length(N)
        m = zeros(n)
        s = zeros(n)

        @showprogress for (i, n) in zip(eachindex(N), N)
            Random.seed!(2022)
            tmp = [generate_data(n, f(n), params, method)[1] for _ in 1:repeats]
            m[i] = @pipe tmp |> median(_)
            s[i] = @pipe tmp |> std(0.25 .* _)
        end
        return m, s
    end
end

eps = 1
# f1 = (; f=n -> log(n), name="log(n)")
# f2 = (; f=n -> log(n) / log(log(n)), name="log(n) / log(log(n))")
# f3 = (; f=n -> log(log(n)), name="log(log(n))")
# f4 = (; f=n -> log(log(n)), name="log(log(n))")
f4 = (; f=n -> eps, name="constant")
F = [f4]

theme(:default)
N = [1000:1000:10_000...]
params = (; p=0.8, r=0.1, clust=3, d=3, order=1, q=Inf, ribbon=true)

begin
    sparse = Any[]
    for f in F
        m, s = one_sim(f.f, N, params, 10, :sparse)
        push!(sparse, [m, s])
        GC.gc()
    end
end;


begin
    plt_sparse = plot(0, 0, xlabel="n", la=0, label="Sparse regime")
    for (f, x) in zip(F, eachrow(sparse))
        plt_sparse = plot(plt_sparse, N, x[1][1], label=f.name, m=:o)
    end
    plt_sparse
    # plot(plt_sparse, ylim=(1e-5, 1), yscale=:identity)
end