using DrWatson
@quickactivate projectdir()

begin
    include(srcdir("rdpg.jl"))
    using Main.rdpg
    using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles, Random
    using ProgressMeter, DataFrames
    using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Distances, LinearAlgebra, UMAP, TSne
    using SparseArrays
end



function read_graph(; path, delim='\t', labels=nothing)
    data = Int.(readdlm(path, delim))
    if labels !== nothing
        rownames = labels[:, 1]
        n = length(unique(data))
        indx = map(i -> findall(j -> j == i, rownames), data)
        A = sparse(indx[:, 1], indx[:, 2], Int(1), n, n)
    else
        tmp = unique(data)
        n = length(tmp)
        data .= minimum(tmp) != 1 ? data .+ 1 : data
        A = sparse(data[:, 1], data[:, 2], Int(1), n, n)
    end
    return A |> LightGraphs.LinAlg.symmetrize
end


begin
    dim = 100
    n = 2000
    clean = true
    path_to_graph = "./data/flights-edges.txt"
    path_to_labels = "./data/flights-info.txt"
end


begin
    labels = readdlm(path_to_labels, ',', header=false)
    Adjacency = read_graph(path=path_to_graph, delim=',', labels=nothing)
end



if clean
    subsample_indices = findall(i -> labels[i, 13] ∉ ["", "\\N"], eachindex(labels[:, 13]))
    # subsample_indices = sample(eachindex(langs), n, replace=false)
    Adjacency = Adjacency[subsample_indices, subsample_indices]
    labels = labels[subsample_indices, :]
end


############################################################################

downsample = false

if !downsample

    ind1 = findall(i -> langs[i] ∈ ["FR", "RU", "ZH"], eachindex(langs))
    indx = sample(ind1, min(length(ind1), 2000), replace=false)
    labs = labels[:, :]

    A = copy(Adjacency)
    Xhat, _ = rdpg.spectralEmbed(A, d=dim, scale=false)

else

    ind1 = findall(i -> langs[i] ∈ ["FR", "RU", "ZH"], eachindex(langs))
    indx = sample(eachindex(ind1), min(length(ind1), 2000), replace=false)

    labs = labels[ind1, :]
    A = copy(Adjacency)[ind1, ind1]
    Xhat, _ = rdpg.spectralEmbed(A, d=dim, scale=false)

end



A = copy(Adjacency)
Xhat, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
indx = eachindex(Xhat[:, 1])

############################################################################

Xnh = Xhat[indx, :]

embedding_umap_x = umap(
    Xnh', 2; n_neighbors=25, metric=Euclidean(),
    neg_sample_rate = 10,
)'
plt_umap_x = scatter(
    embedding_umap_x |> rdpg._Matrix_to_ArrayOfTuples,
    ma=0.5, ms=3, msw=0.01, legend=:bottomleft, #lim=(-12, 12),
    # group=labels[indx, 13],
    hover = labels[indx, 5],
    # marker_z=labels[indx, 12],
    marker_z=norm.(eachrow(labels[:, 9:10])) .* labels[indx, 12],
    title="ϵ = ∞"
)
savefig(plotsdir("flights/plt_umap_x.svg"))


embedding_tsne_x = tsne(Xnh, 2, 0, 2000, 20.0)
plt_tsne_x = scatter(
    embedding_tsne_x |> rdpg._Matrix_to_ArrayOfTuples,
    ms=3, legend=:bottomleft,
    marker_z=norm.(eachrow(labels[:, 9:10])) .* labels[indx, 12],
    # group=labs[indx, 8]
)
savefig(plotsdir("flights/2plt_tsne_x.svg"))


############################################################################

ϵ = 1.0 * log(n)

begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Yhat, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
end

Ynh = Yhat#[indx, :]


embedding_umap_y1 = umap(Ynh', 2; n_neighbors=25, metric=Euclidean())'
plt_umap_y1 = scatter(
    embedding_umap_y1 |> rdpg._Matrix_to_ArrayOfTuples,
    ma=0.5, ms=3, msw=0.01, legend=:bottomleft, #lim=(-12, 12),
    # group=labels[indx, 13],
    hover = labels[indx, 5],
    marker_z=norm.(eachrow(labels[:, 9:10])) .* labels[indx, 12],
    title="ϵ ≍ √log(n)"
)
savefig(plotsdir("flights/2embedding_umap_y1.svg"))




embedding_umap_y1 = umap(Ynh', 2; n_neighbors=25, metric=Euclidean())'
plt_umap_y1 = scatter(
    embedding_umap_y1 |> rdpg._Matrix_to_ArrayOfTuples,
    ms=3, legend=:bottomleft, lim=(-12, 12),
    group=labs[indx, 8],
    title="ϵ ≍ √log(n)"
)
savefig(plotsdir("twitch/2embedding_umap_y1.svg"))





embedding_tsne_y1 = tsne(Ynh, 2, 0, 2000, 20.0)
plt_tsne_y1 = scatter(
    embedding_tsne_y1 |> rdpg._Matrix_to_ArrayOfTuples,
    ms=3, legend=:bottomleft,
    group=labs[indx, 8]
)



begin
    B = nothing
    Yhat = nothing
    GC.gc()
end

############################################################################

ϵ = 1.0 * log(log(n))

begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Yhat, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
end


Ynh = Yhat[indx, :]

embedding_umap_y2 = umap(Ynh', 2; n_neighbors=25, metric=Euclidean())'
plt_umap_y2 = scatter(
    embedding_umap_y2 |> rdpg._Matrix_to_ArrayOfTuples,
    ma=0.5, ms=3, msw=0.01, legend=:bottomleft, #lim=(-12, 12),
    # group=labels[indx, 13],
    hover = labels[indx, 5],
    marker_z=norm.(eachrow(labels[:, 9:10])) .* labels[indx, 12],
    title="ϵ ≍ log(log(n))"
)
savefig(plotsdir("twitch/2embedding_umap_y2.svg"))


embedding_tsne_y2 = tsne(Ynh, 2, 0, 2000, 20.0)
plt_tsne_y2 = scatter(
    embedding_tsne_y2 |> rdpg._Matrix_to_ArrayOfTuples,
    ms=3, legend=:bottomleft,
    group=labs[indx, 8]
)


plot(plt_umap_x, plt_umap_y1, plt_umap_y2, size=(900, 300), layout=(1, 3))
savefig(plotsdir("twitch/embedding_umap_combined.svg"))