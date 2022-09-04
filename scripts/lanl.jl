using DrWatson
@quickactivate projectdir()

begin
    include(srcdir("rdpg.jl"))
    using Main.rdpg
    using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles, Random
    using ProgressMeter, DataFrames
    using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Distances, LinearAlgebra, UMAP, TSne
end

begin
    file = "/storage/home/suv87/work/julia/dp-rdpg/data/tmpdata/shortflows.txt"
    data = readdlm(file, ',', header=false)
    edgelist = data[:, [3, 5]]
    protocol = data[:, [3, 5, 7]]
    writedlm("/storage/home/suv87/work/julia/dp-rdpg/data/tmpdata/shortflows-edgelist.txt", edgelist)
    writedlm("/storage/home/suv87/work/julia/dp-rdpg/data/tmpdata/shortflows-protocol.txt", protocol)
end


begin
    graph_file = "./data/tmpdata/shortflows-edgelist.txt"
    data_file = "./data/tmpdata/shortflows-protocol.txt"
    G = Graphs.loadgraph(graph_file, "graph_key", EdgeListFormat())
    edgelist = readdlm(graph_file, '\t', header=false)
    data = readdlm(data_file, ',', header=false)
end

dim = 100
A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize
Xhat, _ = rdpg.spectralEmbed(A, d=dim, scale=false)


indx = sample(1:size(Xhat, 1), 1000, replace=false)

begin
    Xnh = Xhat[indx, :]
    embedding1 = umap(
        Xnh', 2;
        n_neighbors=20,
        metric=Euclidean(),
        n_epochs=500,
        learning_rate=1,
        min_dist=0.1,
        neg_sample_rate=20
    )'
end

plt2 = scatter(embedding1[:, :] |> rdpg._Matrix_to_ArrayOfTuples, 
        ms=2, ma=0.99, msw=0,
        # group=string.(labels[indx, 2]) .* string.(labels[indx, 8]) .* string.(labels[indx, end])
)



embedding2 = tsne(Xnh, 2, 0, 2000, 10.0)

plt3 = scatter(embedding2[:, :] |> rdpg._Matrix_to_ArrayOfTuples,
    ms=2, ma=0.99, msw=0, label=nothing
    # group=string.(labels[indx, 2]) .* string.(labels[indx, 8]) .* string.(labels[indx, end])
)


#

ϵ = 5.0 * log(n)

begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Ynh, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
    embedding2 = umap(Ynh', 2; n_neighbors=100, metric=Euclidean())'
end

plt3 = @pipe embedding2[:, :] |>
             rdpg._Matrix_to_ArrayOfTuples |>
             scatter(_, ms=2, ma=1, msw=0)




#

ϵ = 5.0 * log(log(n))

begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Ynh, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
    embedding2 = umap(Ynh', 2; n_neighbors=100, metric=Euclidean())'
end

plt4 = @pipe embedding2[:, :] |>
             rdpg._Matrix_to_ArrayOfTuples |>
             scatter(_, ms=2, ma=1, msw=0)






plot(plt2, plt3, plt4, size=(900, 300), layout=(1, 3))