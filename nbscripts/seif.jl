begin
    using DrWatson
    include(srcdir("rdpg.jl"))
    import Main.rdpg
    using LinearAlgebra, Plots, ProgressMeter, Random
    using PersistenceDiagrams, Ripserer, Statistics, StatsBase
end

begin
    function scale_embeddings(X)
        return StatsBase.standardize(ZScoreTransform, X, dims=1)
    end

    function diagram(X; dim_max)
        dgm = ripserer(X |> Alpha, dim_max=dim_max)
        [replace(x -> death(x) == Inf ? PersistenceInterval(birth(x), threshold(d)) : x, d) for d in dgm]
    end

    function bottleneck_distance(Dx, Dy; order=nothing, p=Inf)
        order = isnothing(order) ? 0 : order
        dx, dy = Dx[1+order], Dy[1+order]
        m = max(0, min(length.((dx, dy))...) .- 2)
        dx = dx[end-m:end]
        dy = dy[end-m:end]
        return norm(map((x, y) -> (x .- y) .|> abs |> maximum, dx, dy), p)
    end

    # function bottleneck_distance(Dx, Dy; order=nothing, p=Inf)
    #     Bottleneck()(Dx, Dy)
    # end

    function subsample(X, a=1)
        sample(X |> rdpg.m2t, round(Int, size(X, 1)^a), replace=false)
    end
end


begin
    function generate_sbm_sparse(n, k, p, r)
        f = (x, y) -> (r + p * (x == y)) * log(n) / n
        Z = rand(1:k, n)
        return rdpg.Adjacency(f, Z)
    end

    function generate_sbm_dense(n, k, p, r)
        f = (x, y) -> (r + p * (x == y))
        Z = rand(1:k, n)
        return rdpg.Adjacency(f, Z)
    end

    function simulate_one(n, ϵ, params, type=:dense)
        if type == :dense
            A = generate_sbm_dense(n, params.clust, params.p, params.r)
        else
            A = generate_sbm_sparse(n, params.clust, params.p, params.r)
        end
        X, _, _ = rdpg.spectralEmbed(A, d=params.d)
        Dx = diagram(X |> subsample, dim_max=params.order)

        B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)^2) ./ rdpg.σ(ϵ)^2
        Y, _ = rdpg.spectralEmbed(B, d=params.d)
        Dy = diagram(Y |> subsample, dim_max=params.order)

        # println((:bottleneck, bottleneck_distance(Dx, Dy, order=order, p=p))

        return bottleneck_distance(Dx, Dy, order=params.order, p=params.q)
    end
end



Ks = [1 / 2, 2 / 3, 3 / 4]
# N = [500:250:1000...; 2000:1000:5000]
N = [250:250:1000...]

params = (; p=0.5, r=0.1, clust=3, d=2, order=0, q=Inf)


function simulation(Ks, N; params, repeats=5)

    μ = zeros(length(Ks), length(N))
    σ = zeros(length(Ks), length(N))

    @showprogress for (i, k) in zip(eachindex(Ks), Ks)
        μ[i, :], σ[i, :] = one_sim_k(k, N, params, repeats)
    end

    return μ, σ
end

function one_sim_k(k, N, params, repeats=5, type=:dense)
    n = length(N)
    m = zeros(n)
    s = zeros(n)

    @showprogress for (i, n) in zip(eachindex(N), N)
        ϵn = log(n)^k
        Random.seed!(2022)
        tmp = [simulate_one(n, ϵn, params, type)[1] for _ in 1:repeats]
        m[i] = @pipe tmp |> mean(_)
        s[i] = @pipe tmp |> std(0.25 .* _)
    end
    return m, s
end

a, b = simulation(Ks, N, params=params, repeats=10)
plt = plot(0, 0)
for (x, y) in zip(eachrow(a), eachrow(b))
    plt = plot(plt, x, ribbon=0.5 .* y)
end
plt


begin
    repeats = 5
    Ks = [1 / 2, 2 / 3, 3 / 4]
    Ks_legend = ["0.50", "0.66", "0.75"]
    N = [100, 200, 400, 600, 800, 1000, 2000, 5000]
end

begin
    p, r = 0.5, 0.1
    clust = 3
    d = 2
    n = length(N)
end

begin
    results_dense = [zeros(repeats, n) for _ in 1:length(Ks)]

    prog = Progress(convert(Int, n * repeats * length(Ks)))

    for i in 1:n
        for j in 1:repeats
            for k in eachindex(Ks)
                A = generate_sbm_dense(N[i], clust, p, r)
                ϵn = log(N[i])^(Ks[k])
                error = simulate_one(A, d, ϵn, 1, Inf)
                results_dense[k][j, i] = error[1]
                next!(prog)
            end
        end
    end
end