using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
using Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles
using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
using UMAP, Distances


###############################

begin
    dim = 100
    n = 1000
    ϵ = 5.0 * log(n)
    subsample = false
    path_to_graph = datadir("email-Eu-core.txt")
    path_to_labels = datadir("email-Eu-core-department-labels.txt")
end


begin
    G = Graphs.loadgraph(path_to_graph, "graph_key", EdgeListFormat())
    A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize
    labels = convert.(Int, readdlm(path_to_labels))[:, 2]

    if (subsample)
        N = size(A, 1)
        idx = sample(1:N, n, replace=false)
        A = A[idx, idx]
        labels = labels[idx]
    end
end

begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    # plt1 = @pipe Xnh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3)
end


begin
    embedding = umap(Xnh', 2; n_neighbors=3, metric=Euclidean())'
    # plt2 = @pipe embedding[:, 1:2] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3)
end

plt2 = @pipe embedding[:, :] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, group=labels, ma=0.7, msw=0)