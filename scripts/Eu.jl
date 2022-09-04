using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
using Main.rdpg
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
    n = 1000
    ϵ = 5.0 * log(n)
    subsample = true
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