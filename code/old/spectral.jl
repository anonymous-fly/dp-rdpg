using LinearAlgebra, SparseArrays, Arpack
using JuMP, Gurobi
using LightGraphs, Clustering, Distances
using Random, StatsBase, Statistics, Distributions
using GraphPlot, Plots, LaTeXStrings, ProgressBars

# LinAlg Functions

function Adjacency(f, Z)
    n = size(Z, 1)
    # X = sparse(zeros(n,n))

    if size(Z, 2) == 1
        X = sparse([i > j ? 1 * rand(Bernoulli(f(Z[i], Z[j]))) : 0 for i = 1:n, j = 1:n])
    else
        X = sparse([i > j ? 1 * rand(Bernoulli(f(Z[i, j]))) : 0 for i = 1:n, j = 1:n])
    end

    return LightGraphs.LinAlg.symmetrize(X)
end;


function SpectralEmbed(A; d = 3)
    λ, v = eigs(A, nev = d, maxiter=500)
    X = v * diagm(.√ abs.(λ))
    return X, λ, v
end


function cluster_embeddings(X, d)
    iter = 25
    best = -1
    clusters = nothing

    for i = 1:iter
        res = kmeans(X', d)
        metric = mean(silhouettes(res, pairwise(Euclidean(), X, dims=1)))

        if metric > best
            best = metric
            clusters = res.assignments
        end
    end

    return clusters
end





function privacy(; ϵ::Real = -1, p::Real = -1)
    if ϵ < 0
        return log((1 - p) / p)

    elseif p < 0
        return 1 / (1 + exp(ϵ))
    end
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



function graphplt(A, Z; α = 0.1, scheme = :viridis)
    zcolors = cgrad(scheme, length(unique(Z)), rev = true, categorical = true)[Z]

    gplot(SimpleGraph(A), nodefillc = zcolors, edgestrokec = coloralpha(colorant"grey", α))

end;


function spectralplt(X, Z; α = 1, scheme = :viridis)
    zcolors = cgrad(scheme, length(unique(Z)), rev = true, categorical = true)[Z]
    plt = plot(X[:, 1], X[:, 2], X[:, 3], seriestype = :scatter, label = false, c = zcolors)
    display(plt)
    return plt
end;


# Mechanisms


# Symmetric Edge Flip

_flipSingleEdge(x, p) = rand(Bernoulli(p)) ? 1 - x : x


function edgeFlip(A; ϵ::Real = 1, p::Real = -1, parameters=nothing)
    if p < 0
        p = privacy(ϵ = ϵ)
    end

    n = size(A, 1)
    X = sparse([i>j ? _flipSingleEdge(A[i, j], p) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end;



# Laplace Edge Flip

_LapFlipEdge(x, ϵ) = x + rand(Laplace(0, 1/ϵ)) > 0.5 ? 1 : 0

function laplaceFlip(A; ϵ::Real = 1, parameters=nothing)

    n = size(A, 1)
    X = sparse([i>j ? _LapFlipEdge(A[i, j], ϵ) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end;



# Select-Measure-Reconstruct-type mechanism

_extractEdges(A) = [A[i, j] for i in 1:size(A, 1), j in 1:size(A, 1) if i>j]

function SMR_Mechanism(A; ϵ::Real = 1, parameters=(0.5, 0.5, 100))
    p, q, K  = parameters
    ϵ1 = p * ϵ
    ϵ2 = q * ϵ / K
    n = size(A, 1)

    x = _extractEdges(A)

    M = sum(x) + rand(Laplace(0, ϵ1))

    Σ = []
    for i in 1:K
        push!(Σ, _extractEdges(edgeFlip(A; ϵ=ϵ2)))
    end

   x̂ = SMR_Reconstruct(Σ, M, K)

   Â = sparse(zeros(n,n))

   for i in 1:(n-1)
       for j in (i+1):n
           k = convert(Int, (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
           Â[i, j] = x̂[k]
       end
   end
   return LightGraphs.LinAlg.symmetrize(Â)
end


function SMR_Reconstruct(Σ, M, K)
    model = Model(Gurobi.Optimizer);
    n = size(Σ[1],1)
    @variable(model, z[1:n], Bin)
    @constraint(model, M-1 ≤ sum(z) ≤ M+1)
    # @constraint(model, sum(z) > 0)
    obj = 0
    for i in 1:K
        obj = obj + sum(t -> t^2, z - Σ[i])
    end

    @objective(model, Min, obj)
    optimize!(model)
    return value.(z) .> 0
end


# Composite Functions

function Evaluate(A, Z; M=edgeFlip, ϵ=Inf, parameters=nothing, d=3, plt=false)

    B = M(A; ϵ=ϵ, parameters=parameters);


    X, λ, v = SpectralEmbed(B);

    if plt
        spectralplt(X, Z)
    end

    Clust = cluster_embeddings(X, d);

    println("\n ################################ \n")
    println("Summary: ϵ = $ϵ, |A| = $(sum(A))")
    Accuracy = clustering_accuracy(Clust, Z);
    println("Clustering Accuracy = $Accuracy")

    RelDensity = relativeDensity(A, B);
    println("Relative Density = $RelDensity")

    AdjError = sum(abs.(B - A)) / 2;
    println("Adjacency Discrepancy = $AdjError")
    println()

    return Dict("accuracy" => Accuracy,
        "density" => RelDensity,
        "adjerror" => AdjError)
end



# Example

n = 100
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = Adjacency(f, Z);

# No Privacy
Evaluate(A, Z)

# edgeFlip
Evaluate(A, Z; M=edgeFlip, ϵ=1)

#laplaceFlip
Evaluate(A, Z; M=laplaceFlip, ϵ=1)

#Select-Measure-Reconstruct
SMR_parameters = (0.5, 0.5, 100)
Evaluate(A, Z; M=SMR_Mechanism, ϵ=1, parameters=SMR_parameters)

#Select-Measure-Reconstruct
SMR_Mechanism = (0.2, 0.8, 100)
Evaluate(A, Z; M=DN_Mechanism, ϵ=1, parameters=SMR_parameters)



# Evaluation

ϵ = 0.5:0.5:2
Repeat = 5

μ_eflip = zeros(length(ϵ),3);
σ_eflip = zeros(length(ϵ),3);

μ_lflip = zeros(length(ϵ),3);
σ_lflip = zeros(length(ϵ),3);

μ_smr1 = zeros(length(ϵ), 3);
σ_smr1 = zeros(length(ϵ), 3);

μ_smr2 = zeros(length(ϵ), 3);
σ_smr2 = zeros(length(ϵ), 3);

μ_smr3 = zeros(length(ϵ), 3);
σ_smr3 = zeros(length(ϵ), 3);


SMR_parameters1 = (0.5, 0.5, 100)
SMR_parameters2 = (0.5, 0.5, 500)
SMR_parameters3 = (0.2, 0.8, 100)

eflip = []
lflip = []
smr1 = []
smr2 = []
smr3 = []

for i = tqdm(1:Repeat)

    n = 100
    p, q, r = 0.4, 0.1, 0.15
    f = (x, y) -> r + p * (x == y) - q * (x != y)
    Z = rand([1, 2, 3], n);
    A = Adjacency(f, Z);

    push!(eflip , []); eflip[i] = []
    push!(lflip , []); lflip[i] = []
    push!(smr1 ,  []); smr1[i]  = []
    push!(smr2 ,  []); smr2[i]  = []
    push!(smr3 ,  []); smr3[i]  = []

    for j = tqdm(1:length(ϵ))
        push!(eflip[i], Evaluate(A, Z; M=edgeFlip, ϵ=ϵ[j]));
        push!(lflip[i], Evaluate(A, Z; M=laplaceFlip, ϵ=ϵ[j]));
        push!(smr1[i], Evaluate(A, Z; M=SMR_Mechanism, ϵ=ϵ[j], parameters=SMR_parameters1));
        push!(smr2[i], Evaluate(A, Z; M=SMR_Mechanism, ϵ=ϵ[j], parameters=SMR_parameters2));
        push!(smr3[i], Evaluate(A, Z; M=SMR_Mechanism, ϵ=ϵ[j], parameters=SMR_parameters3));
    end
end

metrics = ["accuracy", "density", "adjerror"]

for i = 1:length(metrics)
    for j in 1:length(E)

        μ_eflip[j, i] = mean([eflip[u][j][metrics[i]] for u in 1:Repeat])
        σ_eflip[j, i] = std([eflip[u][j][metrics[i]] for u in 1:Repeat])
        μ_lflip[j, i] = mean([lflip[u][j][metrics[i]] for u in 1:Repeat])
        σ_lflip[j, i] = std([lflip[u][j][metrics[i]] for u in 1:Repeat])
        μ_smr1[j, i]  = mean([smr1[u][j][metrics[i]]  for u in 1:Repeat])
        σ_smr1[j, i]  = std([smr1[u][j][metrics[i]]  for u in 1:Repeat])
        μ_smr2[j, i]  = mean([smr2[u][j][metrics[i]]  for u in 1:Repeat])
        σ_smr2[j, i]  = std([smr2[u][j][metrics[i]]  for u in 1:Repeat])
        μ_smr3[j, i]  = mean([smr3[u][j][metrics[i]]  for u in 1:Repeat])
        σ_smr3[j, i]  = std([smr3[u][j][metrics[i]]  for u in 1:Repeat])

    end
end

lwd = 3
fillalpha = 0.2

for i in 1:length(metrics)
    plt = plot(E, μ_eflip[:,i], ribbon=0.1*μ_eflip[:,i], label=uppercasefirst("EdgeFlip"), linewidth=lwd, fillalpha=fillalpha)
    title!(uppercasefirst(metrics[i]))
    plot!(E, μ_lflip[:,i], ribbon = 0.1*μ_lflip[:,i], label=uppercasefirst("LaplaceFlip"), linewidth=lwd, fillalpha=fillalpha)
    plot!(E, μ_smr1[:,i], ribbon = 0.1*μ_smr1[:,i], label=uppercasefirst("SMR1"), linewidth=lwd, fillalpha=fillalpha)
    plot!(E, μ_smr2[:,i], ribbon = 0.1*μ_smr2[:,i], label=uppercasefirst("SMR2"), linewidth=lwd, fillalpha=fillalpha)
    plot!(E, μ_smr3[:,i], ribbon = 0.1*μ_smr3[:,i], label=uppercasefirst("SMR3"), linewidth=lwd, fillalpha=fillalpha)
    display(plt)
end
