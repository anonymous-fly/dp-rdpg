using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
using Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles
using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
using TSne, UMAP, Distances
using SparseArrays


function read_graph(; path, delim='\t', labels=nothing)
    data = Int.(readdlm(path, delim))
    if labels !== nothing
        rownames = labels[:, 1]
        tmp = unique(rownames)
        indx = map(i -> findall(j -> j == i, rownames), data)
        A = spzeros(Int, length(tmp), length(tmp))
        # A[CartesianIndex.(eachcol(indx)...)] .= Int(1)
        for i in eachrow(indx)
            A[i...] = Int(1)
            # println(i)
        end
    else
        tmp = unique(data)
        data .= minimum(tmp) != 1 ? data .+ 1 : data
        A = spzeros(Int, length(tmp), length(tmp))
        # A[CartesianIndex.(eachcol(data)...)] .= Int(1)
        for i in eachrow(data)
            A[i...] = Int(1)
        end
    end
    return A |> LightGraphs.LinAlg.symmetrize
end


begin
    dim = 100
    n = 1000
    ϵ = 5.0 * log(n)
    subsample = false
    path_to_graph = "./data/tmpdata/cora-edges.txt"
    path_to_labels = "./data/tmpdata/cora-labels.txt"
end


begin
    labels = convert.(Int, readdlm(path_to_labels))
    A = read_graph(path=path_to_graph, delim=',', labels=nothing)
    if (subsample)
        N = size(A, 1)
        idx = sample(1:N, n, replace=false)
        A = A[idx, idx]
        labels = labels[idx, :]
    end
    labels = labels[:, 2]
end


begin
    # remove = [1, 2, 27, 117, 69, 85, 118]
    remove = [164]
    indx = map(x -> x ∉ remove, 1:size(A, 1))
    A = A[indx, indx]
    labels = labels[indx]
end


begin
    degs = [sum(A, dims=2)...]
    deg_range = range(quantile(degs, [0.05, 0.95])...)
    filtered_vertices = findall(i -> (degs[i] ∈ deg_range), eachindex(degs))
    A = A[filtered_vertices, filtered_vertices]
    labels = labels[filtered_vertices]
end



begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    plt1 = @pipe Xnh[:, 1:2] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, legend=nothing, hover=1:size(Xnh, 1))
end

embedding1 = umap(Xnh', 2; n_neighbors=20, metric=Euclidean())'
scatter(embedding1 |> rdpg._Matrix_to_ArrayOfTuples, markersize=3, group=labels, hover=1:length(labels))

embedding2 = tsne(Xnh, 2, 0, 500, 50.0)
scatter(embedding2 |> rdpg._Matrix_to_ArrayOfTuples, markersize=3, group=labels, hover=1:length(labels))

Dist = pairwise(Euclidean(), Xnh, dims=1)
Dist .= Dist + 1e-10 * ones(size(Dist)) - diagm(repeat([1e-10], size(Dist, 1)))

heatmap(Dist)

Dx = ripserer(Dist, dim_max=1, reps=true)
plot(Dx)

x = dgmclust(Dx; order=1)

Dx[1][end-6]

