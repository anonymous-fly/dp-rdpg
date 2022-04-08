# LinAlg Functions

function Adjacency(f, Z)
    n = size(Z, 1)
    # X = sparse(zeros(n,n))

    if size(Z, 2) == 1
        X = sparse([i > j ? 1 * rand(Bernoulli(f(Z[i], Z[j]))) : 0 for i = 1:n, j = 1:n])
    else
        X = sparse([i > j ? 1 * rand(Bernoulli(min( f(Z[i, j]), 1))) : 0 for i = 1:n, j = 1:n])
    end

    return LightGraphs.LinAlg.symmetrize(X)
end;


function SpectralEmbed(A; d = 3, scale=true)

    λ, v = eigs(A, nev = d, maxiter=500)
    X = v * diagm(.√ abs.(λ))
    
    if scale
        X = StatsBase.standardize(ZScoreTransform, X, dims=1)
    end

    return X, λ, v
end



function cluster_embeddings(X, d)
    iter = 25
    best = -1
    clusters = nothing

    for i = 1:iter
        res = kmeans(X', d)
        metric = mean(silhouettes(res, pairwise(Distances.Euclidean(), X, dims=1)))

        if metric > best
            best = metric
            clusters = res.assignments
        end
    end

    return clusters
end




# Some Helper Functions

_Matrix_to_ArrayOfTuples = M -> tuple.(eachcol(M)...)


function privacy(; ϵ::Real = -1, p::Real = -1)
    if ϵ < 0
        return log((1 - p) / p)

    elseif p < 0
        return 1 / (1 + exp(ϵ))
    end
end



# TDA Functions

function plt3d(X; camera=(30,30))
    if size(X[1]) != ()
        plt = scatter(X[1], X[2], X[3], 
        camera = camera, aspect_ratio=:equal, legend=false,
        markersize=1)
        display(plt)
    else
        plt = scatter(X[:, 1], X[:, 2], X[:, 3], 
        camera = camera, aspect_ratio=:equal, legend=false,
        markersize=1)
        display(plt)
    end
    return(plt)
end;


function _TorusToEuclidean(p; R, r)
    θ, ϕ = p
    x = (R .+ (r .* cos.(θ))) .* cos.(ϕ)
    y = (R .+ (r .* cos.(θ))) .* sin.(ϕ)
    z = r .* sin.(θ)
    return x, y, z
end


function randTorus(n; d=2, R=2, r=0.5, euclidean=false)
    T = Torus(d)
    
    if euclidean
        X = [_TorusToEuclidean(random_point(T), R=R, r=r) for i in 1:n]
    else
        X = [tuple(random_point(T)...) for i in 1:n]
    end

    return X
end


function randSphere(n; d=2)
    S = Sphere(d)
    X = [tuple(random_point(S)...) for i in 1:n]
    return X
end



_lemniscate_gerono_coord = x -> asin(x[1])
lemniscate_geodesic = (x, y) -> abs(asin(x[1]) - asin(y[1]))
lemniscate_distance = (x, y) ->  norm(x .- y)


function randLemniscate(n; σ = 0)
    
    signal = R"tdaunif::sample_lemniscate_gerono($n)"
    noise = R"matrix(rnorm(2 * $n, 1, $σ), ncol=2)"
    
    X = tuple.(eachcol(rcopy(signal + noise))...)
    
    return X
end



_log_transform_interval(x, minval) = PersistenceInterval( tuple(max.(log.(x), fill(minval, 2))...) )

function log_transform_diagram(D)
    logD = deepcopy(D)
    minval = log(maximum(D[1][1]))
    for i in 1:length(logD)
        logD[i] = PersistenceDiagram(_log_transform_interval.(logD[i], minval), dim=i-1)
    end
    return logD
end


function diagram(X; d=2, log_trans=false, alpha=true)

    points = tuple.(eachcol(X)...)
    
    if alpha
        dgm = ripserer(Alpha(points), dim_max=d)
    else
        dgm = ripserer(EdgeCollapsedRips(points), dim_max=d)
    end

    return log_trans ? log_transform_diagram(dgm) : dgm
end


function bottleneck_distance(X, Y; log_trans=false, plt_scatter=false, plt_diagram=false, pers_dim=nothing)
    
    DX = diagram(X, log_trans=log_trans)
    DY = diagram(Y, log_trans=log_trans)

    plts = []
    
    if plt_scatter
        push!(plts, scatter(_Matrix_to_ArrayOfTuples(X), label=false, color=:dodgerblue, title="Original"))
        push!(plts, scatter(_Matrix_to_ArrayOfTuples(Y), label=false, color=:firebrick1, title="Private ϵ=$ϵ"))
    end

    if plt_diagram
        push!(plts, plot(DX, title="Original"))            
        push!(plts, plot(DY, title="Private ϵ=$ϵ"))
    end

    if plt_diagram || plt_scatter
        display(plot(plts..., layout = length(plts)))
    end

    return Bottleneck()(DX[pers_dim + 1], DY[pers_dim + 1])
end


# Some utilities



function clustering_accuracy(ξ, Z)
    # return 1 - Clustering.varinfo(ξ, Z)
    return randindex(ξ, Z)[2]
end



function relativeDensity(A, B)
    # return (sum(B) - sum(A))/sum(A)
    return sum(B)/sum(A)
end



function graphplt(A; Z = nothing, α = 0.1, scheme = :viridis, color = "red")
    if Z !== nothing
        zcolors = cgrad(scheme, length(unique(Z)), rev = true, categorical = true)[Z]
        gplot(SimpleGraph(A), nodefillc = zcolors, edgestrokec = coloralpha(colorant"grey", α))
    else
        gplot(SimpleGraph(A), nodefillc = parse(Colorant, color), edgestrokec = coloralpha(colorant"grey", α))
    end
end;


function spectralplt(X; Z = nothing, α = 1, scheme = :viridis, color = "red")
    if Z !== nothing
        zcolors = cgrad(scheme, length(unique(Z)), rev = true, categorical = true)[Z]
        plt = plot(X[:, 1], X[:, 2], X[:, 3], seriestype = :scatter, label = false, c = zcolors)
        display(plt)
    else
        plt = plot(X[:, 1], X[:, 2], X[:, 3], seriestype = :scatter, label = false, c = parse(Colorant, color))
        display(plt)
    end
        
    return plt
end;


# Composite Functions

function Evaluate(A, Z; M=edgeFlip, d=3, sbm=true, plt=false, log_trans=true, pers_dim=1, kwargs...)

    B = M(A; kwargs...);
    Xᵇ, _ = SpectralEmbed(A)
    X, λ, v = SpectralEmbed(B);

    results = Dict()

    push!(results, "density" => relativeDensity(A, B));
    push!(results, "adjerror" => sum(abs.(B - A)) / 2);
    
    push!(results, 
    "bottleneck" => bottleneck_distance(Xᵇ, X, log_trans=log_trans, plt_scatter=plt, plt_diagram=plt, pers_dim=pers_dim))

    if sbm
        Clust = cluster_embeddings(X, d);
        Accuracy = clustering_accuracy(Clust, Z);
        push!(results, "accuracy" => Accuracy)
    end

    println("\n\n################################ \n")
    println("""
    Summary: \n \n 
    Mechanism = $M
    $(tuple.(kwargs...))
    |A| = $(sum(A)) 
    sbm = $sbm \n""")

    for (key, value) in results
        println("$key = $value")
    end

    return results
end



function simulation(f, Z;
    M = edgeFlip, 
    ϵ = Inf, 
    runs=10, 
    parameters=nothing, 
    d=3,
    metrics = ["density", "bottleneck", "adjerror"],
    kwargs...
    )

    μ = zeros(length(ϵ),length(metrics));
    σ = zeros(length(ϵ),length(metrics));
    results = []

    for i = tqdm(1:runs)
        A = Adjacency(f, Z);
        push!(results , []); results[i] = []
        for j = tqdm(1:length(ϵ))
            push!(results[i], Evaluate(A, Z; M=M, ϵ=ϵ[j], kwargs...));
        end
    end

    for i = 1:length(metrics)
        for j in 1:length(ϵ)
            μ[j, i] = mean([results[u][j][metrics[i]] for u in 1:runs])
            σ[j, i] = std([results[u][j][metrics[i]] for u in 1:runs])
        end
    end

    return μ, σ, results

end


function plot_results(vals, result_list, legends; 
    lwd=3, fillalpha=0.2, ribbon_scale=0.9,bottleneck_scale=0.9,
    metrics = ["bottleneck", "density", "adjerror"]
    )

    # i = findall(x -> x=="accuracy", metrics)[1]
    plts = []
    for i = 1:length(metrics)

        if metrics[i]=="bottleneck"
            scale = bottleneck_scale
        else
            scale = ribbon_scale
        end

        subplt = plot(vals, result_list[1][1][:, i],
        ribbon= scale * result_list[1][2][:, i],
        label = uppercasefirst(legends[1]),
        linewidth=lwd, fillalpha=fillalpha, legend=:bottomright)
        title!(subplt, uppercasefirst(metrics[i]))

        for j in 2:length(result_list)
            plot!(subplt, vals, result_list[j][1][:, i],
            ribbon=ribbon_scale * result_list[j][2][:, i],
            label = uppercasefirst(legends[j]),
            linewidth=lwd, fillalpha=fillalpha)
        end

        display(subplt)
        push!(plts, subplt)
    end
    return plts
end;
