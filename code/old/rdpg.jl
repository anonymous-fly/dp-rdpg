using LinearAlgebra, SparseArrays, Arpack
using JuMP, Gurobi
using LightGraphs, Clustering, Distances
using Random, StatsBase, Statistics, Distributions
using GraphPlot, Plots, LaTeXStrings, ProgressBars
using HomotopyContinuation, Suppressor
using Manifolds, Manopt, Ripserer, PersistenceDiagrams
using Rotations
import Manifolds: distance
import Distances: Euclidean

#= 
RCall is used to call `tdaunif::sample_lemniscate_gerono` in R, 
which samples points uniformly from a Lemniscate. Make sure you run:
        $ install.packages("tdaunif")
in your R session before running the code for the Lemniscate in Julia.
 =#
using RCall


include("mechanisms.jl")
include("spectral-functions.jl")
include("tomato.jl")



#=  1. Stochastic Block Model =#

# 1A: Load Data
n=200
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = Adjacency(f, Z);


# 1B. Demo
X, _ = SpectralEmbed(A, scale=true)
spectralplt(X, Z = Z) # Scatterplot

D = diagram(X, log_trans=true) # Get persistence diagram from Alpha complex
plot(D, markeralpha=0.5) # plot log-persistence diagram. 
plot!([-100, 0, 0], [0, 0, 100], label=false, c=:plum) # You should see 3 blue dots separated from the rest of the pack


# 1C. Private version
ϵ = 1.5
B = edgeFlip(A, ϵ=ϵ)
Y, _ = SpectralEmbed(B, scale=true)
bottleneck_distance(X, Y, plt_diagram=true, plt_scatter=true, log_trans=true)

# 1D. Combined
Evaluate(A, Z, M=edgeFlip, plt=true, sbm=true, log_trans=true, ϵ = ϵ)

plotly() # Alternatively you can look at it using Plotly backend
Evaluate(A, Z, M=edgeFlip, plt=true, sbm=true, log_trans=true, ϵ = 1)








#= 2. Sociability Network =#

# plotly() # Use plotly backend if you're interested in examining the 3d plots

# 2A. Generate Data
n = 1000
Z = rand(Gamma(1, 1), n)
f = (x, y) -> 1 - exp(-2 * x * y)
A = Adjacency(f, Z);

# 2B. Demo
X, _ = SpectralEmbed(A, scale=false)
plt3d(X) # The data is generated from a 1-dim manifold, and this is recovered here.

# 2C. Persistence Diagram
D = diagram(X, log_trans=true)
plot(D)
plot!([-100, 0, 0], [0, 0, 100], linewidth=2, c=:plum, label=false)


# 2D. Combined
Evaluate(A, Z, M=edgeFlip, plt=true, sbm=false, log_trans=true, ϵ=2)









#= 3. Circle =#
# In this example, the latent manifold underlying the graph is a circle. 

# 3A. Generate Data
σ = 0.5  # Bandwidth for Gaussian (trace-class) kernel
d = 1
M = Sphere(d) # Define the circle using the Manifolds package
Z = randSphere(n, d=d)
f = (x, y) -> min(1, pdf(Normal(0, σ), distance(M, x, y)))
A = Adjacency(f, Z);

# 3B. Scatterplot
X, _ = SpectralEmbed(A)
plt3d(X)

# 3C. Persistence Diagram
D = diagram(X, log_trans=true)
plot(D)
plot!([-100, 0, 0],[0, 0,100], linewidth=2, c=:plum, label=false)


# 3D. Combined
Evaluate(A, Z, M=edgeFlip, plt=true, sbm=false, log_trans=true, ϵ=2)
Evaluate(A, Z, M=edgeFlip, plt=true, sbm=false, log_trans=true, ϵ=1)








#= 4. Lemniscate =#
# The next example uses the Lemniscate of Gerono
# It's plotted here for your convenience


plot(t -> 1 + sin(t), t -> 1 + sin(t) * cos(t), 0, 2π, label="lemniscate", linewidth=2)
scatter!(randLemniscate(100), label="Without Noise")
scatter!(randLemniscate(100, σ=0.05), label="With noise")

σ = 0.75  # Bandwidth
Z = randLemniscate(n)
f = (x, y) -> min(1, pdf(Normal(0, σ), lemniscate_distance(x, y)))
A = Adjacency(f, Z);

X, _ = SpectralEmbed(A)
plt3d(X)

D = diagram(X, log_trans=true)
plot(D)
plot!([-100, 0, 0],[0, 0,100], linewidth=2, c=:plum, label=false)

Evaluate(A, Z, M=edgeFlip, plt=true, sbm=false, log_trans=true, ϵ=2)
Evaluate(A, Z, M=edgeFlip, plt=true, sbm=false, log_trans=true, ϵ=1)



# 5. Sphere

σ = 0.6  # Bandwidth for Gaussian (trace-class) kernel
d = 2
M = Sphere(d) # Define the circle using the Manifolds package
Z = randSphere(n, d=d)
f = (x, y) -> min(1, pdf(Normal(0, σ), distance(M, x, y)))
A = Adjacency(f, Z);

# Embed in 4-dim space
X, _ = SpectralEmbed(A, d=4)

# Persistence Diagram
D = diagram(X, log_trans=true)
plot(D)



# 5. Torus

σ = 1  # Bandwidth for Gaussian (trace-class) kernel
d = 2
M = Torus(d) # Define the circle using the Manifolds package
Z = randTorus(n, d=d, euclidean=false)
f = (x, y) -> min(1, pdf(Normal(0, σ), distance(M, [x...], [y...])))
A = Adjacency(f, Z);

# Embed in 4-dim space
X, _ = SpectralEmbed(A, d=4)

# Persistence Diagram
D = diagram(X, log_trans=true)
plot(D)

print("Hi")