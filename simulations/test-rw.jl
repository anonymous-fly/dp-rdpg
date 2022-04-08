import Pkg;
Pkg.activate(".")


include("../../code/src/rdpg.jl")
import Main.rdpg
using Graphs, GraphIO, LightGraphs, StatsBase, Pipe, Ripserer, LightGraphs, TSne

# path_to_graph = "/storage/home/suv87/work/julia/grdpg/code/datasets/com-amazon.ungraph.txt"
# path_to_graph = "/storage/home/suv87/work/julia/grdpg/code/datasets/loc-brightkite_edges.txt"
path_to_graph = "/storage/work/s/suv87/julia/grdpg/code/datasets/email-Eu-core.txt"
# path_to_graph = "/storage/home/suv87/work/julia/grdpg/code/datasets/M2Anonymized.csv"
# path_to_graph = "/storage/home/suv87/work/julia/grdpg/code/datasets/PGPgiantcomponent.txt"
G = Graphs.loadgraph(path_to_graph, "graph_key", EdgeListFormat())
A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize

N = size(A)[1]
n = 1005
idx = sample(1:N, n, replace = false)

Xnh, _ = rdpg.spectralEmbed(A, d = 10)
@pipe Xnh[idx, 1:3] |> rdpg.spectralplt(_)

# @pipe Xnh |> StatsBase.standardize(ZScoreTransform, _, dims = 1) |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=1)
D = ripserer(rdpg._Matrix_to_ArrayOfTuples(Xnh[idx, :]), sparse = true, dim_max = 1)
plot(D)


Y = tsne(Xnh[idx, :], 3, 100, 1000, 10.0)
@pipe Y |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize = 1)
D = ripserer(Alpha(rdpg._Matrix_to_ArrayOfTuples(Y)), dim_max = 2)
plot([D[1]] |> rdpg.log_transform_diagram)
logD = rdpg.log_transform_diagram(D)
plot(logD)



Y1 = tsne(Xnh[idx, :], 3, 100, 1000, 10.0)
Y2 = tsne(Xnh[idx, :], 3, 100, 1000, 10.0)
D1 = ripserer(Alpha(rdpg._Matrix_to_ArrayOfTuples(Y1)), dim_max = 2)
D2 = ripserer(Alpha(rdpg._Matrix_to_ArrayOfTuples(Y2)), dim_max = 2)
Bottleneck()(D1, D2)