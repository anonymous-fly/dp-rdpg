using DrWatson
@quickactivate projectdir()

begin
    include(srcdir("rdpg.jl"))
    using Main.rdpg
    using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles
    using ProgressMeter, Dates
    using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Distances, LinearAlgebra, UMAP
    using SparseArrays
end


function read_graph(; path, delim='\t', labels=nothing)
    data = Int.(readdlm(path, delim))
    if labels !== nothing
        rownames = labels[:, 1]
        tmp = unique(rownames)
        indx = map(i -> findall(j -> j == i, rownames), data)
        A = spzeros(Int, length(tmp), length(tmp))
        # A[CartesianIndex.(eachcol(indx)...)] .= Int(1)
        for i in eachrow(indx)
            # A[i...] = Int(1)
            println(i)
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
    n = 2000
    ϵ = 5.0 * log(n)^2
    subsample = true
    path_to_graph = datadir("loc-brightkite_edges.txt")
    path_to_labels = datadir("loc-brightkite_totalCheckins.txt")
end


begin
    Adjacency = read_graph(path=path_to_graph, delim='\t', labels=nothing)
    labels = readdlm(path_to_labels, '\t')
    labels[:, 2] = Date.(labels[:, 2], dateformat"y-m-dTH:M:SZ")
end


begin
    A = Adjacency
    ids = copy(labels)
    degs = [sum(A, dims=2)...]
    deg_range = range(quantile(degs, [0.25, 0.75])...)
    filtered_vertices = findall(i -> (degs[i] ∈ deg_range), eachindex(degs))
end



begin
    YEAR = [2009]
    MONTH = [6]

    active_id_indices = findall(
        i -> (year(ids[i, 2]) ∈ YEAR) && (month(ids[i, 2]) ∈ MONTH),
        eachindex(ids[:, 2])
    )
    active_vertices = unique(convert.(Int, ids[active_id_indices, 1])) .+ 1


    active_ids = ids[active_id_indices, :]
    lat = zeros(length(active_vertices))
    long = zeros(length(active_vertices))

    @showprogress for i in eachindex(active_vertices)
        lat[i], long[i] = mean(active_ids[active_ids[:, 1].==active_vertices[i]-1, 3:4], dims=1)
    end

    A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize

    lat_q = quantile(lat, [0.1, 0.9])
    long_q = quantile(long, [0.1, 0.9])

    indx = (lat_q[1] .≤ lat .≤ lat_q[2]) .&& (long_q[1] .≤ long .≤ long_q[2])

    active_vertices = active_vertices[indx]
    lat = lat[indx]
    long = long[indx]
    A = A[active_vertices, active_vertices]


    if subsample
        idx = sample(1:size(A, 2), n, replace=false)
        A = A[idx, idx]
        lat = lat[idx]
        long = long[idx]
    else
        idx = 1:size(A, 2)
    end

end


begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    embedding = umap(Xnh', 3; n_neighbors=10, metric=Euclidean())'
end


plt = scatter(embedding[:, 1:2] |> rdpg._Matrix_to_ArrayOfTuples, markersize=2, marker_z=long+lat)

#


begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Ynh, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
    embedding2 = umap(Ynh', 3; n_neighbors=100, metric=Euclidean())'
end

plt3 = @pipe embedding2[:, :] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, marker_z=norm.(eachrow(ids[:, 3:4])), ms=3, ma=0.5, msw=0, label=nothing)


plot(plt2, plt3, size=(900, 400))