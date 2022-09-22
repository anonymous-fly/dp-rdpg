# rdpg.jl

module rdpg

using ArnoldiMethod
# using KrylovKit
using Distances
using Distributions
using Clustering
using LightGraphs
using Pipe
using Plots
using Ripserer
using Ripserer: PersistenceInterval, birth, death, persistence, Alpha, ripserer
using SparseArrays
using Statistics
using StatsBase
using Random
using LinearAlgebra

import Base: log

export _Matrix_to_ArrayOfTuples,
    _ArrayOfTuples_to_Matrix,
    _ArrayOfVectors_to_ArrayOfTuples,
    _ArrayOfTuples_to_ArrayOfVectors,
    m2t,
    t2m,
    v2t,
    t2v,
    subsample,
    bottleneck_distance,
    _log_transform_interval,
    _flipSingleEdge,
    V,
    read_network,
    Adjacency,
    generate_sbm,
    privacy,
    σ,
    τ,
    edgeFlip,
    spectralEmbed,
    scale_embeddings,
    spectralplt,
    graphplt,
    cluster_embeddings,
    diagram,
    scale,
    log_transform_diagram,
    randCircle,
    randLemniscate,
    randSphere,
    _LapFlipEdge,
    laplaceFlip

include("helper-functions.jl")
include("plot-functions.jl")
include("rdpg-functions.jl")
include("spectral-functions.jl")
include("tda-functions.jl")



end
