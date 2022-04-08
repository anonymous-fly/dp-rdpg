ENV["GKSwstype"]="100"

import Pkg; Pkg.activate(pwd() * "/../../"); Pkg.instantiate();
using Ripserer, PersistenceDiagrams, Plots, Statistics, ProgressBars
include("./networks.jl")


function generate_sbm(n, k, p, r)
    f = (x, y) -> r + p * (x == y)
    Z = rand(1:k, n)
    return generate_rdpg(f, Z)
end

function diagram(X, dim_max; alpha=true)
    points = tuple.(eachcol(X)...)
    dgm = ripserer(Alpha(points), dim_max=dim_max)
    return dgm
end

function bottleneck_distances(X, Y, dim_max)
    DX = diagram(X, dim_max)
    DY = diagram(Y, dim_max)
    return [Bottleneck()(DX[d], DY[d]) for d in 1:dim_max+1]
end

function scale_embeddings(X)
    # c = cov(X)
    # U = eigvecs(c)
    # s = U * Diagonal(eigvals(c) .^ -0.5) * transpose(U)
    return ( X .- mean(eachrow(X))' ) * ( X'X )^(-0.5)
end

function simulate_one(A, d, epsilon, method)
    # Note: we add one to d, so don't add one yourself!
    X, _, _ = spectral_embeddings(A, d = d+1, scale = false)
    
    A_private = edge_flip(A, ϵ = epsilon)
    
    if method == :eps
        A_private = A_private .- privacy(ϵ = epsilon)
    end
    
    X_private, _, _ = spectral_embeddings(A_private, d = d+1, scale = false)
    
    if method == :eps
        X_private = X_private ./ (1 - 2 * privacy(ϵ = epsilon))
    elseif method == :noeps
        X = scale_embeddings(X)
        X_private = scale_embeddings(X_private)
        # X = StatsBase.standardize(ZScoreTransform, X, dims=1)
        # X_private = StatsBase.standardize(ZScoreTransform, X_private, dims=1)
    end
    
    # return maximum(bottleneck_distances(X, X_private, d+1))
    # why not record all dimensions and combine later?
    return bottleneck_distances(X, X_private, d)
end


function V(V, fun=mean; slice=1, i=1)
    if slice==1
        reshape( fun( V, dims=1 ), size(V, 3), size(V, 2), : )[i,:]
    elseif slice==2
        reshape( fun( V, dims=1 ), size(V, 3), size(V, 2), : )[:,i]
    end
end

function generate_sbm(n, k, p, r)
    f = (x, y) -> r + p * (x == y)
    Z = rand(1:k, n)
    return generate_rdpg(f, Z)
end

println("The current working directory is ")
println(pwd())
println(pwd() * "/plots/")

p, r = 0.4, 0.1, 0.15
clust = 3

N = [50, 100, 200, 400, 600]
ϵ = [0.5, 1, 2, 4, 10]
# ϵ = [0.5, 1]

n = length(N)
m = length(ϵ)
repeats = 20

ne1 = zeros(repeats, n, m)
we1 = zeros(repeats, n, m)

ne2 = zeros(repeats, n, m)
we2 = zeros(repeats, n, m)

ne3 = zeros(repeats, n, m)
we3 = zeros(repeats, n, m)

for i in tqdm(1:n)
    for j in 1:m
        for k in 1:repeats
            A = generate_sbm(N[i], clust, p, r)
            for method in [:eps, :noeps]
                results = simulate_one(A, 2, ϵ[j], method)
                # fields = [n, ϵ, method]
                # append!(fields, results)
                # println(join(fields, "\t"))
                if method==:eps
                    we1[k, i, j] = results[1];
                    we2[k, i, j] = results[2];
                    we3[k, i, j] = results[3];
                else
                    ne1[k, i, j] = results[1];
                    ne2[k, i, j] = results[2];
                    ne3[k, i, j] = results[3];                
                end
            end
        end
    end
end




plt1 = plot(title="ϵ publicly available")
i=1; plot!(plt1, N, V(we1, mean; slice=1, i=i), ribbon=V(we1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
i=2; plot!(plt1, N, V(we1, mean; slice=1, i=i), ribbon=V(we1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
i=3; plot!(plt1, N, V(we1, mean; slice=1, i=i), ribbon=V(we1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
# i=4; plot!(plt1, N, V(we1, mean; slice=1, i=i), ribbon=V(we1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
savefig(plt1, "./plots/sbm-witheps.pdf")

plt2 = plot(title="ϵ not publicly available")
i=1; plot!(plt2, N, V(ne1, mean; slice=1, i=i), ribbon=V(ne1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
i=2; plot!(plt2, N, V(ne1, mean; slice=1, i=i), ribbon=V(ne1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
i=3; plot!(plt2, N, V(ne1, mean; slice=1, i=i), ribbon=V(ne1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
# i=4; plot!(plt2, N, V(ne1, mean; slice=1, i=i), ribbon=V(ne1, std; slice=1, i=i), label="ϵ = $(ϵ[i])")
savefig(plt2, "./plots/sbm-noeps.pdf")
