import Pkg;
Pkg.activate(".");

include("../src/rdpg.jl")
import Main.rdpg
using Distributions, Pipe



# Stochastic Blockmodel
# SBM
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = rdpg.Adjacency(f, Z);
X, λ = rdpg.spectralEmbed(A; d = 3)
plt1 = scatter(rdpg._Matrix_to_ArrayOfTuples(X),
    marker_z = Z, ratio = 1, markersize = 5,
    colorbar=false, label=nothing,
    xlim = (-3, 3), ylim = (-3, 3), zlim = (-3, 3))

B = rdpg.edgeFlip(A; ϵ = 2)
Y, η = rdpg.spectralEmbed(B; d = 3)
plt2 = scatter(rdpg._Matrix_to_ArrayOfTuples(Y),
    marker_z = Z, ratio = 1, markersize = 5,
    colorbar=false, label=nothing,
    xlim = (-3, 3), ylim = (-3, 3), zlim = (-3, 3))

plot(plt1, plt2, size=(900, 400))