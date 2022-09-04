using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
using Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles
using ProgressMeter
using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
using Distances, LinearAlgebra, UMAP



begin
    function scale(x)
        n = size(x, 1)
        return x * sqrt(x' * (diagm(ones(n)) - (1 / n .* ones(n) * ones(n)')) * x) ./ sqrt(n)
    end

    function stdscore(w)
        return [abs(x - mean(w)) / std(w) for x in w]
    end

    function extract_vertices(d, f=Ripserer.representative)
        return @pipe d |> f .|> Ripserer.vertices |> map(x -> [x...], _) |> rdpg.t2m |> unique
    end

    function filterDgm(dgm; order=1, ς=3, f=Ripserer.representative, vertex=true)

        if order == 1
            u = Ripserer.persistence.(dgm[order][1:end-1])
            v = [0, u[1:end-1]...]
        else
            u = Ripserer.persistence.(dgm[order][1:end])
            v = [0, u[1:end-1]...]
        end

        # w = u - v
        w = u
        index = findall(x -> x > ς, stdscore(w))
        return vertex ? [extract_vertices(dgm[order][i...], f) for i in index] : index
    end

    function dgmclust(dgm; order=1, threshold=3)
        idx = filterDgm(dgm, order=order, ς=threshold)
        K = length(idx)
        classes = repeat([0], length(dgm[1]))
        for k in K
            classes[idx[k]] .= k
        end
        return classes
    end
end

begin
    function diagram(X;)
        points = tuple.(eachcol(X)...)
        dgm = ripserer(points, dim_max=0)
        return dgm
    end

    function bottleneck_distances(X, Y, dim_max)
        DX = diagram(X, dim_max)
        DY = diagram(Y, dim_max)
        return [Bottleneck()(DX[1], DY[1])]
    end

    function simulate_one(A, d, epsilon)
        X, _, _ = rdpg.spectralEmbed(A, d=d + 1, scale=false)
        A_private = rdpg.edgeFlip(A, ϵ=epsilon) .- rdpg.privacy(ϵ=epsilon)
        X_private, _, _ = rdpg.spectralEmbed(A_private, d=d + 1, scale=false)
        X_private = X_private ./ (1 - 2 * rdpg.privacy(ϵ=epsilon))
        return bottleneck_distances(X, X_private, d)
    end
end



begin
    dim = 20
    n = 2000
    ϵ = 5.0 * log(n)^2
    subsample = true
    path_to_graph = datadir("loc-brightkite_edges.txt")
    path_to_labels = datadir("loc-brightkite_totalCheckins.txt")
end



begin
    G = Graphs.loadgraph(path_to_graph, "graph_key", EdgeListFormat())
    A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize
    labels = readdlm(path_to_labels, '\t')
    # labels[:, 2] = Date.(labels[:, 2], dateformat"y-m-dTH:M:SZ")
    # ids = copy(labels)

    degs = [sum(A, dims=2)...]
    deg_range = range(quantile(degs, [0.25, 0.75])...)
    filtered_vertices = findall(i -> (degs[i] ∈ deg_range), eachindex(degs))
end

begin

    YEAR = [2009]
    MONTH = [2]

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
    end

end


begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    embedding = umap(Xnh', 3; n_neighbors=50, metric=Euclidean())'
end


plt2 = @pipe embedding[:, :] |>
             rdpg._Matrix_to_ArrayOfTuples |>
             scatter(
                 _, ms=1, ma=0.5, msw=0, label=nothing,
                 marker_z=long
             )

#


begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Ynh, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
    embedding2 = umap(Ynh', 3; n_neighbors=100, metric=Euclidean())'
end

plt3 = @pipe embedding2[:, :] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, marker_z=norm.(eachrow(ids[:, 3:4])), ms=3, ma=0.5, msw=0, label=nothing)


plot(plt2, plt3, size=(900, 400))