using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
import Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs, DelimitedFiles
using Plots, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase



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
    dim = 20
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
    end
end


begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    plt1 = @pipe Xnh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=3, groups=labels, legend=nothing)
end

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
    k = 3
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
    topological_clusters = repeat([0], size(Xnh, 1))

    idx = filterDgm(Dx, order=1, ς=3)

    for k in 1:length(idx)
        topological_clusters[idx[k]] .= k
    end

    randindex(topological_clusters, filtered_labels)
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
    u1 = Ripserer.persistence.(Dy[order][1:end-1])
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
    private_topological_clusters = repeat([0], size(Ynh, 1))

    idy = filterDgm(Dy, order=1, ς=1)

    for k in 1:length(idy)
        private_topological_clusters[idy[k]] .= k
    end

    randindex(private_topological_clusters, filtered_labels)
end


Bottleneck()(Dx, Dy)



one_sim = 