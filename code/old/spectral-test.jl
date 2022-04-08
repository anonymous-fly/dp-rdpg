using LinearAlgebra, SparseArrays, Arpack
using JuMP, Gurobi
using LightGraphs, Clustering, Distances
using Random, StatsBase, Statistics, Distributions
using GraphPlot, Plots, LaTeXStrings, ProgressBars
using HomotopyContinuation, Suppressor

include("mechanisms.jl")
include("spectral-functions.jl")


# Example

n = 100
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = Adjacency(f, Z);

# No Privacy
Evaluate(A, Z; M=identity)

# edgeFlip
Evaluate(A, Z; M=edgeFlip, ϵ=1)

#laplaceFlip
Evaluate(A, Z; M=laplaceFlip, ϵ=1)

#Select-Measure-Reconstruct
SMR_parameters = (0.5, 0.5, 100)
Evaluate(A, Z; M=SMR_Mechanism, ϵ=1, parameters=SMR_parameters)

#Select-Measure-Reconstruct
SMR_parameters = (0.2, 0.8, 100)
Evaluate(A, Z; M=SMR_Mechanism, ϵ=1, parameters=SMR_parameters)

# Johnson-Lindenstrauss
Evaluate(A, Z; M=JLMechanism, ϵ = .01, δ = 0.001)

#Sparse Mechanism
Sparse_parameters = (0.01, 0.99)
Evaluate(A, Z; M=SparseMechanism, ϵ=1, parameters=Sparse_parameters)

# LowRankReconstructMechanism
# `parameters` is the rank of the approximation you want to use
Evaluate(A, Z; M=LowRankReconstructMechanism, ϵ = 1, dims = 3)


# Evaluation

metrics = ["accuracy", "density", "bottleneck", "adjerror"]

ϵ = 0.5:0.5:3
runs = 10

μ_eflip, σ_eflip = simulation(f, Z; M=edgeFlip, ϵ=ϵ, runs=runs, metrics=metrics);
# μ_lflip, σ_lflip = simulation(f, Z; M=laplaceFlip, ϵ=ϵ, runs=runs, metrics=metrics);
μ_lflip, σ_lflip = simulation(f, Z; M=JLMechanism, ϵ=ϵ, runs=runs, metrics=metrics, δ=1e-5);

result_list = [(μ_eflip, σ_eflip), (μ_lflip, σ_lflip)];
result_legends = ["edgeFlip", "laplaceFlip"]

plot_results(ϵ, result_list, result_legends, metrics=metrics)



SMR_parameters = (0.1, 0.9, 100)
μ_smr, σ_smr = simulation(f, Z; M=SMR_Mechanism, ϵ=ϵ, runs=runs, parameters=SMR_parameters);
push!(result_list, (μ_smr, σ_smr))
push!(result_legends, "SMR")

plts = plot_results(ϵ, result_list, result_legends)


Sparse_parameters = (0.01, 0.99)
μ_sparse, σ_sparse = simulation(f, Z; M=SparseMechanism, ϵ=ϵ, runs=runs, parameters=Sparse_parameters);

push!(result_list, (μ_sparse, σ_sparse))
push!(result_legends, "Sparse")

plts = plot_results(ϵ, result_list, result_legends)



graphplt(A,Z = Z)
B = sparsify(A; l=0)
