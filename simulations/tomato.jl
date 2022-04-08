import Pkg;
Pkg.activate(".")

# include("../../code/src/rdpg.jl")
include("../../code/src/rdpg.jl")
import Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, ProgressBars
using LinearAlgebra, Distances, Manifolds, Manopt, Distributions
using Plots, TSne, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase


function extract_vertices(d, f = Ripserer.representative)
    return @pipe d |> f .|> Ripserer.vertices |> map(x -> [x...], _) |> rdpg.t2m |> unique
end

function stdscore(w)
    return [abs(x - mean(w)) / std(w) for x in w]
end

function filterDgm(dgm; order = 1, f = Ripserer.representative)
    u = Ripserer.persistence.(dgm[order][1:end-1])
    v = [0, u[1:end-1]...]
    w = u - v
    index = findall(x -> x > 2.2, stdscore(w))
    return [extract_vertices(dgm[order][i...], f) for i in index]
end

function dgmclust(dgm; order = 1)
    idx = filterDgm(dgm, order = order)
    K = length(idx)
    classes = repeat([0], length(dgm[1]))
    for k in K
        classes[idx[k]] .= k
    end
    return classes
end


begin
    Random.seed!(2021)
    n = 300
    d = 1
    M = Sphere(d) # Define the circle using the Manifolds package
    μ = [0, 1, 0]  # repeat([0.5], 3)
    Z1 = [tuple(([random_point(M)...] .* 0.9)...) for i in 1:(3*n)]
    Z2 = [tuple(([random_point(M)...] .* 0.5)...) for i in 1:n]
    Z = [Z1; Z2]
    scatter(Z, label = false, aspect_ratio = :equal, title = "Latent Space")
    σ = 0.5  # Bandwidth for Gaussian (trace-class) kernel
    f = (x, y) -> abs.((dot([x...], [y...])))
    A = rdpg.Adjacency(f, Z)
    X, _ = rdpg.spectralEmbed(A, scale = false, d = 3)

    true_class = [repeat([0], 3 * n); repeat([1], n)]
end

Ms = (X, ϵ) -> (X .- rdpg.τ(ϵ)^2) ./ rdpg.σ(ϵ)^2

begin
    ϵ = 1
    B = rdpg.edgeFlip(A, ϵ = ϵ)
    Y, _ = @pipe B |> Ms(_, ϵ) |> rdpg.spectralEmbed(_, scale = false, d = 3)
end


D = @pipe Y |> rdpg.m2t |> ripserer(Alpha(_), dim_max = 1, reps = true)
C1 = rdpg.cluster_embeddings(Y, 2) .- 1;
C2 = repeat([0], size(Y, 1))
idx = filterDgm(D, order = 1)
for k in 1:length(idx)
    C2[idx[k]] .= k
end


truth = @pipe Z |>
              scatter(_, ratio = 1, camera = (75, 30), lim = (-2, 2), legend = :bottomleft,
    group = true_class, title = "Latent Space", size = (300, 300))

kmeans = @pipe Y |>
               rdpg.scale |>
               rdpg.m2t |>
               scatter(_, ratio = 1, camera = (75, 30), lim = (-2, 2), legend = :bottomleft,
                   group = C1, title = L"$k-$Means clustering (ϵ=1)", size = (300, 300))

# C2 = tomatoPlot(X, graph = 2, k1 = 0.2, k2 = 3, tao = 500, plt = false);
topological = @pipe Y |>
                    rdpg.scale |>
                    rdpg.m2t |>
                    scatter(_, ratio = 1, camera = (75, 30), lim = (-2, 2), legend = :bottomleft,
                        group = C2, title = "Topological clustering (ϵ=1)", size = (300, 300))

savefig(plot(topological, size = (250, 250)), "./plots/tclust.pdf")
savefig(plot(kmeans, size = (250, 250)), "./plots/kmeans.pdf")
savefig(plot(truth, size = (250, 250)), "./plots/true.pdf")




function one_sim(ϵ)
    B = rdpg.edgeFlip(A, ϵ = ϵ)
    Y, _ = @pipe B |> Ms(_, ϵ) |> rdpg.spectralEmbed(_, scale = false, d = 3)
    D = @pipe Y |> rdpg.m2t |> ripserer(Alpha(_), dim_max = 1, reps = true)
    C1 = rdpg.cluster_embeddings(Y, 2) .- 1
    C2 = repeat([0], size(Y, 1))
    idx = filterDgm(D, order = 1)
    for k in 1:length(idx)
        C2[idx[k]] .= k
    end
    c1 = mean(abs.(C1 .!= true_class))
    c2 = mean(abs.(C2 .!= true_class))
    return (c1, c2)
end


reps = 10
E = [0.5, 1, 2, 5]

n = reps
m = length(E)

U = zeros(n, m)
V = zeros(n, m)

for j in tqdm(1:m)
    for i in tqdm(1:n)
        res = one_sim(E[j])
        U[i, j] = res[1]
        V[i, j] = res[2]
    end
end


df = zeros(m*n*2,3)

for k in 1:2
    for j in 1:m
        for i in 1:n
            df[i + n*(j-1) ,1] = U[i,j]
            df[i + n*(j-1) ,2] = E[j]
            df[i + n*(j-1) ,3] = 1
            df[i + n*(j-1) + n*m*(k-1),1] = V[i, j]
            df[i + n*(j-1) + n*m*(k-1),2] = E[j]
            df[i + n*(j-1) + n*m*(k-1),3] = 2
        end
    end
end