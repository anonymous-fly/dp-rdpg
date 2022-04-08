using LinearAlgebra, SparseArrays, Arpack
using JuMP, Gurobi
using LightGraphs, Clustering, Distances
using Random, StatsBase, Statistics, Distributions
using GraphPlot, Plots, LaTeXStrings, ProgressBars

include("spectral-functions.jl")


n = 100
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = Adjacency(f, Z);

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
    plt = plot(E, μ_eflip[:,i], ribbon=0.5*σ_eflip[:,i], label=uppercasefirst("EdgeFlip"), linewidth=lwd, fillalpha=fillalpha)
    title!(uppercasefirst(metrics[i]))
    plot!(E, μ_lflip[:,i], ribbon = 0.5*σ_lflip[:,i], label=uppercasefirst("LaplaceFlip"), linewidth=lwd, fillalpha=fillalpha)
    plot!(E, μ_smr1[:,i], ribbon = 0.5*σ_smr1[:,i], label=uppercasefirst("SMR1"), linewidth=lwd, fillalpha=fillalpha)
    plot!(E, μ_smr2[:,i], ribbon = 0.5*σ_smr2[:,i], label=uppercasefirst("SMR2"), linewidth=lwd, fillalpha=fillalpha)
    plot!(E, μ_smr3[:,i], ribbon = 0.5*σ_smr3[:,i], label=uppercasefirst("SMR3"), linewidth=lwd, fillalpha=fillalpha)
    display(plt)
    savefig(uppercasefirst(metrics[i])*".pdf")
end
