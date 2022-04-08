using LinearAlgebra, SparseArrays, Arpack
using JuMP, Gurobi
using LightGraphs, Clustering, Distances
using Random, StatsBase, Statistics, Distributions
using GraphPlot, Plots, LaTeXStrings, ProgressBars
using HomotopyContinuation, Suppressor
using Manifolds, Manopt, Ripserer, PersistenceDiagrams
import Manifolds: distance

#= 
RCall is used to call `tdaunif::sample_lemniscate_gerono` in R, 
which samples points uniformly from a Lemniscate. Make sure you run:
        $ install.packages("tdaunif")
in your R session before running the code for the Lemniscate in Julia.
 =#
using RCall


include("mechanisms.jl")
include("spectral-functions.jl")

n = 100
σ = 0.75  # Bandwidth
Z = randLemniscate(n)
f = (x, y) -> min(1, pdf(Normal(0, σ), lemniscate_distance(x, y)))
A = Adjacency(f, Z);


metrics = ["density", "bottleneck", "adjerror"]

ϵ = 0.5:0.2:10
runs = 10

μ_eflip, σ_eflip = simulation(f, Z; M=edgeFlip, ϵ=ϵ, runs=runs, metrics=metrics, sbm=false, pers_dim=1);
μ_lflip, σ_lflip = simulation(f, Z; M=laplaceFlip, ϵ=ϵ, runs=runs, metrics=metrics, sbm=false, pers_dim=1);
Sparse_parameters = (0.01, 0.99)
μ_sparse, σ_sparse = simulation(f, Z; M=SparseMechanism, ϵ=ϵ, runs=runs, parameters=Sparse_parameters, sbm=false, pers_dim=1);

result_list = [(μ_eflip, σ_eflip), (μ_lflip, σ_lflip), (μ_sparse, σ_sparse)] ;
# result_list = [(μ_eflip, σ_eflip), (μ_lflip, σ_lflip)] ;
result_legends = ["edgeFlip", "laplaceFlip", "Sparse"]

plot_results(ϵ, result_list, result_legends, metrics=metrics, bottleneck_scale=0.5)




