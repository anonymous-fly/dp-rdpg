# rdpg.jl

module rdpg

using Arpack
using Distances
using Distributions
using Clustering
using GraphIO
using GraphPlot
using IJulia
using LightGraphs
using LinearAlgebra
using Manifolds
using Manopt
using PersistenceDiagrams
using PersistenceDiagramsBase
using Pipe
using Plots
using Ripserer
using Setfield
using SparseArrays
using Statistics
using StatsBase
using LinearAlgebra
using Pkg
using Random
using RCall

import Base: log

export _Matrix_to_ArrayOfTuples,
    _ArrayOfTuples_to_Matrix,
    _ArrayOfVectors_to_ArrayOfTuples,
    _ArrayOfTuples_to_ArrayOfVectors,
    m2t,
    t2m,
    v2t,
    t2v,
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

include("structures.jl")
include("helper-functions.jl")
include("plot-functions.jl")
include("rdpg-functions.jl")
include("spectral-functions.jl")
include("tda-functions.jl")



end
