include("../../code/src/rdpg.jl")
import Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, ProgressBars
using LinearAlgebra, Distances, Manifolds, Manopt, Distributions
using Plots, TSne, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase

using Cairo, Compose

Random.seed!(2021);

n = 300;

d = 1;
M = Sphere(d); # Define the circle using the Manifolds package
μ = [0, 1, 0]  # repeat([0.5], 3)
Z1 = [tuple(([random_point(M)...] .* 0.9)...) for i in 1:(3*n)]
Z2 = [tuple(([random_point(M)...] .* 0.5)...) for i in 1:n]
Z = [Z1; Z2]
scatter(Z, label = false, aspect_ratio = :equal, title = "Latent Space")

σ = 0.5  # Bandwidth for Gaussian (trace-class) kernel
# f = (x, y) -> min(1, pdf(Normal(0, σ), Distances.Euclidean()([x...], [y...])))
# f = (x, y) -> 0.1 * (dot([x...],[y...])) .^ 2
f = (x, y) -> abs.((dot([x...], [y...])))
A = rdpg.Adjacency(f, Z);
X, _ = rdpg.spectralEmbed(A, scale = false, d = 3)
plotly();
scatter(X |> rdpg.m2t, markersize = 1);
gr();

cam = (80, 40)

C1 = rdpg.cluster_embeddings(X, 2);
P1 = scatter(rdpg._Matrix_to_ArrayOfTuples(X[:, 1:3]), c = C1, aspect_ratio = :equal, camera = cam, label = "")
title!("K-means")
# savefig(P1, "./plots/kmeans.pdf")

C2 = tomatoPlot(X, graph = 2, k1 = 0.2, k2 = 3, tao = 500, plt = false);
P2 = scatter(rdpg._Matrix_to_ArrayOfTuples(X[:, 1:3]), c = C2, aspect_ratio = :equal, camera = cam, label = "")
title!("ToMATo")
# savefig(P2, "./plots/tomato.pdf")

gplot(SimpleGraph(A), edgestrokec = coloralpha(colorant"grey", 0.1), layout = random_layout, linetype = "curve")

ind = sample(1:length(Z), 500)
A_sub = Matrix(A)[ind, ind]

layout = (args...) -> spring_layout(args...; C = 20)
draw(PDF("./plots/networkplot.pdf", 10cm, 10cm),
    gplot(SimpleGraph(A_sub),
        nodefillc = coloralpha(colorant"dodgerblue", 0.9),
        edgestrokec = coloralpha(colorant"black", 0.1),
        layout = layout))



# # Rotate in 3D
# R = rand(RotMatrix{3})
# R = RotXY(π/2,π/2)

# d = 1
# M = Sphere(d) # Define the circle using the Manifolds package
# μ = [0, 1, 0]  # repeat([0.5], 3)
# Z1 = [tuple(([random_point(M)...; 0] .+ μ)...) for i in 1:n]
# Z2 = [tuple(( R * [random_point(M)... ; 0] )...) for i in 1:n]
# Z = [Z1; Z2]
# scatter(Z, label=false)

# σ = 0.6  # Bandwidth for Gaussian (trace-class) kernel
# f = (x, y) -> min(1, pdf(Normal(0, σ), Distances.Euclidean()([x...], [y...])))
# A = Adjacency(f, Z);
# X, _ = SpectralEmbed(A, )
# plotly(); plt3d(X); gr(); 

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
    index = findall(x -> x > 3, stdscore(w))
    return [extract_vertices(dgm[order][i...], f) for i in index]
end

function dgmclust(dgm; order=1)
    idx = filterDgm(dgm, order = order)
    K = length(idx)
    classes = repeat([0], length(dgm[1]))
    for k in K
        classes[idx[k]] .= k
    end
    return classes
end


D = @pipe X |> rdpg.m2t |> ripserer(Alpha(_), dim_max = 1, reps = true)
C3 = dgmclust(D, order=1)
plt = scatter(X |> rdpg.m2t, group=C3, ratio=1, camera=cam, size=(400,400))
title!("ToMATo")
savefig(plt, "./plots/tomato/tomato.pdf")