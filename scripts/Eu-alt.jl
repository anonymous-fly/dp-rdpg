using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
using Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles
using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
using TSne, UMAP, Distances

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
    degs = [sum(A, dims=2)...]
    deg_range = range(quantile(degs, [0.05, 0.95])...)
    filtered_vertices = findall(i -> (degs[i] ∈ deg_range), eachindex(degs))
    A = A[filtered_vertices, filtered_vertices]
    labels = labels[filtered_vertices]
end


begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    plt1 = @pipe Xnh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, legend=nothing)
end


embedding1 = umap(Xnh', 2; n_neighbors=2, metric=Euclidean())'
scatter(embedding1 |> rdpg._Matrix_to_ArrayOfTuples, markersize=3)

embedding2 = tsne(Xnh, 2, 0, 2000, 5.0)
scatter(embedding2 |> rdpg._Matrix_to_ArrayOfTuples, markersize=3)

begin
    Dx = @pipe Xnh |>
               rdpg._Matrix_to_ArrayOfTuples |>
               ripserer(_, dim_max=0)
    plot(Dx)
end

begin
    q = 3
    order = 1 # order = dim + 1
    u1 = Ripserer.persistence.(Dx[order][1:end-1])
    v1 = [0, u1[1:end-1]...]
    # w1 = u1 - v1
    w1 = u1
    index1 = findall(x -> x > q, stdscore(w1))
    println(index1)
    plot(
        plot(Dx), Dx[order][index1], lim=(-0.1, Inf),
        markercolor="red", markeralpha=1, markershape=:x, markersize=7
    ) |> display
end


begin
    k = 20
    freqs = @pipe labels |> countmap |> collect |> sort(_, by=x -> x[2], rev=:true)
    labs = [x[1] for x in freqs[1:k]]
    filtered_labels = filter(x -> x in labs, labels)
    filtered_ids = filter(i -> labels[i] in labs, 1:size(A, 1))
    A = A[filtered_ids, filtered_ids]
end

begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    plt1 = @pipe Xnh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, groups=filtered_labels)
end

begin
    Dx = @pipe Xnh |>
               rdpg._Matrix_to_ArrayOfTuples |>
               ripserer(_, dim_max=1, reps=true, alg=:involuted)
    plot(Dx)
end



begin
    q = 4
    order = 1 # order = dim + 1
    u1 = Ripserer.persistence.(Dx[order][1:end-1])
    index1 = findall(x -> x > q, stdscore(u1))
    println(index1)
    plot(
        plot(Dx), Dx[order][index1], lim=(-0.1, Inf),
        markercolor="red", markeralpha=1, markershape=:x, markersize=7
    ) |> display
end


#


begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    Ynh, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
    plt2 = @pipe Ynh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3)
end

begin
    Dy = @pipe Ynh |>
               rdpg._Matrix_to_ArrayOfTuples |>
               ripserer(_, dim_max=1, reps=true)
    plot(Dy)
end

begin
    q = 2.9
    order = 1 # order = dim + 1
    u2 = Ripserer.persistence.(Dy[order][1:end-1])
    index2 = findall(x -> x > q, stdscore(u2))
    println(index2)
    plot(
        plot(Dy), Dy[order][index2], lim=(-0.1, Inf),
        markercolor="red", markeralpha=1, markershape=:x, markersize=7
    ) |> display
end

begin

    repeats = 5
    Eps = [0.05, 0.1, 0.2, 0.5, 0.8, 1:3...]
    m = length(Eps)

    results = zeros(repeats, m)

    prog = Progress(convert(Int, m * repeats))

    for i in 1:m
        for j in 1:repeats
            error = simulate_one(A, 10, Eps[i])
            results[j, i] = error[1]
            next!(prog)
        end
    end

    plt = plot(title="Email Eu Core Data", xlabel="ϵ", ylabel="Bottleneck Distance", ledend=:right)
    plt = plot(plt, Eps,
        mean(results, dims=1)',
        ribbon=std(results, dims=1)',
        marker=:o,
        label=nothing
    )
    vline!([sqrt(log(n))], position=:bottomright, label="√log(n)", line=:dot, c=:firebrick1)

    # savefig(plot(plt, size=(400, 300)), plotsdir("eu/bottleneck.pdf"))

end




function filt_dgm(D; k=1)

    dt = D[1][end-k].death
    xrips = Rips(Xnh |> rdpg.m2t)
    edges_xrips = Ripserer.edges(xrips)
    vertices_xrips = map(x -> x.birth < dt && Ripserer.vertices(x), edges_xrips)
    vertices_xrips = filter(x -> x != false, vertices_xrips)

    clusters = Any[]
    for i in 1:length(vertices_xrips)
        if length(clusters) == 0
            push!(clusters, [vertices_xrips[i]...])
        else
            index = -1
            for j in 1:length(clusters)
                if (vertices_xrips[i][1] in clusters[j]) || (vertices_xrips[i][2] in clusters[j])
                    index = j
                    break
                end
            end

            if index > 0
                push!(clusters[index], vertices_xrips[i]...)
            else
                push!(clusters, [vertices_xrips[i]...])
            end
        end
    end

    for i in 1:length(clusters)
        clusters[i] = unique(clusters[i])
    end

    return clusters
end

begin
    topological_clusters = repeat([0], size(Xnh, 1))
    idx = filt_dgm(Dx, k=20)
    for k in 1:length(idx)
        topological_clusters[idx[k]] .= k
    end
    randindex(topological_clusters, filtered_labels)[2]
end

begin
    private_topological_clusters = repeat([0], size(Ynh, 1))
    idy = filt_dgm(Dy, k=20)
    for k in 1:length(idx)
        private_topological_clusters[idy[k]] .= k
    end
    randindex(private_topological_clusters, filtered_labels)[2]
end



list1 = [0, 1, 3, 4, 5, 7, 9, 10, 14, 17, 6, 32, 8, 36, 15, 26, 13, 21, 16, 37, 34, 20, 22, 28, 27, 23]
labs = map(x -> x ∈ list1, labels)
scatter(embedding1 |> rdpg._Matrix_to_ArrayOfTuples, markersize=3, groups=labs, hover=labels)

sum(labs), sum(.!labs)
size(list1)