using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
import Main.rdpg
using PersistenceDiagrams, Pipe, Plots, ProgressMeter, Random, Ripserer, Statistics, StatsBase

function diagram(X, dim_max; alpha=false)
    points = tuple.(eachcol(X)...)
    if alpha
        dgm = ripserer(Alpha(points), dim_max=dim_max)
    else
        dgm = ripserer(points, dim_max=dim_max)
    end
    return dgm
end

function bottleneck_distances(X, Y, dim_max, a)
    DX = diagram(X, dim_max; alpha=a)
    DY = diagram(Y, dim_max; alpha=a)
    return [Bottleneck()(DX[d], DY[d]) for d in 1:dim_max+1]
end


function generate_sbm_sparse(n, k, p, r)
    f = (x, y) -> (r + p * (x == y)) * 10 * log(n) / n
    Z = rand(1:k, n)
    return rdpg.Adjacency(f, Z)
end

function generate_sbm_dense(n, k, p, r)
    f = (x, y) -> (r + p * (x == y))
    Z = rand(1:k, n)
    return rdpg.Adjacency(f, Z)
end

function simulate_one(A, X, d, ϵ; a=false)
    X, _, _ = rdpg.spectralEmbed(A, d=d + 1, scale=false)
    A_private = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    X_private, _, _ = rdpg.spectralEmbed(A_private, d=d + 1, scale=false)
    return bottleneck_distances(X, X_private, d, a)
end


function simulate_one(A, D, d, ϵ; a=false)
    A_private = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    X_private, _, _ = rdpg.spectralEmbed(A_private, d=2, scale=false)
    D_private = diagram(X_private, d; alpha=a)
    return [Bottleneck()(D[d], D_private[d]) for d in 1:d+1]
end


diagram_params = (; alpha=true, dim_max=0)

####################################################################################
# Dense Regime

begin
    repeats = 2
    Ks = [2 / 3, 3 / 4, 1]
    # Ks_legend = ["1/2", "2/3", "1"]
    # Ks_legend = ["0.33", "0.66", "0.90"]
    Ks_legend = ["0.66", "0.75", "1.00"]
    N = [1000, 2000]
end



begin
    p, r = 0.5, 0.1
    clust = 3
    n = length(N)
end


begin
    results_dense = [zeros(repeats, n) for _ in 1:length(Ks)]
    prog = Progress(convert(Int, n * repeats * length(Ks)))
    for i in 1:n
        for j in 1:repeats
            A = generate_sbm_dense(N[i], clust, p, r)
            X, _, _ = rdpg.spectralEmbed(A, d=2, scale=false)
            DX = diagram(X, 0; alpha=true)

            for k in 1:length(Ks)

                ϵn = 1 * log(N[i])^(Ks[k])
                error = simulate_one(A, DX, 0, ϵn, a=true)
                results_dense[k][j, i] = error[1]
                next!(prog)
            end
        end
    end
end



theme(:default)
plt_dense = plot(title="ϵ=logᵏ(n)", xlabel="n", ylabel="Bottleneck Distance")
for k in 1:length(Ks)
    plot!(plt_dense, N,
        map(x -> x == Inf ? missing : x, results_dense[k]) |> eachcol .|> skipmissing .|> mean,
        # ribbon=std(results_dense[k], dims=1),
        marker=:o,
        label="k=$(Ks_legend[k])",
        lw=3, fillapha=0.01,
        yformatter=identity
    )
end
plt_dense


####################################################################################
# Sparse Regime


begin
    repeats = 2
    Ks = [1 / 2, 1]
    Ks_legend = ["0.5", "1.00"]
    N = [2000, 5000]
end


begin
    p, r = 0.5, 0.3
    clust = 3
    repeats = 3
    n = length(N)
end


begin
    results_sparse = [zeros(repeats, n) for _ in 1:length(Ks)]
    prog = Progress(convert(Int, n * repeats * length(Ks)))

    for i in 1:n
        for j in 1:repeats
            A = generate_sbm_sparse(N[i], clust, p, r)
            X, _, _ = rdpg.spectralEmbed(A, d=2, scale=false)
            DX = diagram(X, 0; alpha=true)

            for k in 1:length(Ks)

                ϵn = 1 * log(N[i])^(Ks[k])
                error = simulate_one(A, DX, 0, ϵn, a=true)
                results_sparse[k][j, i] = error[1]
                next!(prog)
            end
        end
    end
end


theme(:default)
plt_sparse = plot(title="ϵ=logᵏ(n)", xlabel="n", ylabel="Bottleneck Distance")
for k in 1:length(Ks)
    plot!(plt_sparse, N,
        map(x -> x == Inf ? missing : x, results_sparse[k]) |> eachcol .|> skipmissing .|> mean,
        # ribbon=std(results_3[k], dims=1),
        marker=:o,
        label="k=$(Ks_legend[k])",
        lw=3, fillapha=0.01,
    )
end
plt_sparse





####################################################################################

n = 1000
A = generate_sbm_dense(n, 3, 0.05, 0.02)
X, _, _ = rdpg.spectralEmbed(A, d=3, scale=false)
DX = ripserer(X |> rdpg.m2t, dim_max=0, threshold=1)[1]

A_private = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
X_private, _ = rdpg.spectralEmbed(A_private, d=3, scale=false)
D_private = ripserer(X_private |> rdpg.m2t, dim_max=0, threshold=1)[1]
Bottleneck()(DX, D_private)
