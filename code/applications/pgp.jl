using DrWatson
@quickactivate projectdir()

include(srcdir("rdpg.jl"))
import Main.rdpg
using StatsBase, Pipe, Graphs, GraphIO, LightGraphs
using Plots, TSne, Ripserer, PersistenceDiagrams, PersistenceDiagramsBase

# function scale(x)
#     return (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
# end

function scale(x)
    n = size(x, 1)
    return x * sqrt(x' * (diagm(ones(n)) - (1 / n .* ones(n) * ones(n)')) * x) ./ sqrt(n)
end


begin
    dim = 10
    n = 1000
    ϵ = 5.0 * log(n) / n
    subsample = false
    # subsample = true
    # path_to_graph = "/storage/home/suv87/work/julia/grdpg/code/datasets/PGPgiantcomponent.txt"
    path_to_graph = datadir("email-Eu-core.txt")
    path_to_labels = datadir("email-Eu-core-department-labels.txt")
    # path_to_graph = "/storage/work/s/suv87/julia/grdpg/code/datasets/email-Eu-core.txt"
end

labels = convert.(Int, readdlm(path_to_labels))[:, 2]

begin
    G = Graphs.loadgraph(path_to_graph, "graph_key", EdgeListFormat())
    A = Graphs.LinAlg.adjacency_matrix(G) |> LightGraphs.LinAlg.symmetrize

    if (subsample)
        N = size(A, 1)
        idx = sample(1:N, n, replace=false)
        A = A[idx, idx]
    end
end

begin
    Xnh, _ = rdpg.spectralEmbed(A, d=dim, scale=false)
    # plt1 = @pipe Xnh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=1)
    plt1 = @pipe Xnh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=1)
end

begin
    Dx = @pipe Xnh |>
               #    scale |>
               rdpg._Matrix_to_ArrayOfTuples |>
               #    ripserer(_, sparse = true, dim_max = 1)
               ripserer(Alpha(_), dim_max=1)
    plot(Dx)
end

begin
    B = (rdpg.edgeFlip(A, ϵ=ϵ) .- rdpg.τ(ϵ)) ./ rdpg.σ(ϵ)^2
    # B = rdpg.edgeFlip(A, ϵ = ϵ)
    Ynh, _ = rdpg.spectralEmbed(B, d=dim, scale=false)
    plt2 = @pipe Ynh[:, 1:3] |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, markersize=1)
end

begin
    data = Ynh |>
           scale |>
           rdpg._Matrix_to_ArrayOfTuples
    Dy = @pipe data |>
               ripserer(_, dim_max=1, alg=:involuted, reps=true)
    # ripserer(Alpha(_), dim_max = 1, alg = :involuted, reps = true)
    plot(Dy)
end



id1 = @pipe Dy[1] .|> persistence |> tiedrank |> findall(x -> x >= partialsort(_, 2, rev=true), _)
id2 = @pipe Dy[2] .|> persistence |> tiedrank |> findall(x -> x >= partialsort(_, 2, rev=true), _)


temp = @pipe Dy[2][id2] .|> (death_simplex(_), birth_simplex(_))
temp = @pipe Dy[2][id2] .|> representative
scatter(data, markersize=1)
scatter!(representative(Dy[2][end]), data; label="cycle", markersize=2)
scatter!(representative(Dy[2][end-1]), data; label="cycle", markersize=2)
scatter!(representative(Dy[2][end-2]), data; label="cycle", markersize=2)
scatter!(Dy[2][end-1], data; label="cycle", markersize=2)
scatter!(Dy[1][end-1], data; label="cycle", markersize=2)



# # t-SNE
begin
    Xn = tsne(Xnh, 3, 0, 1000, 20.0)
    Yn = tsne(Ynh, 3, 0, 5000, 10.0)
end

begin
    plt3 = @pipe Xn[:, 1:2] |> scale |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, c=:dodgerblue, markeralpha=0.3)
    # plt3 = @pipe Yn |> rdpg._Matrix_to_ArrayOfTuples |> scatter(_, c = :dodgerblue, markeralpha = 0.3)
    plt3 = @pipe Yn[:, 1:2] |> scale |> rdpg._Matrix_to_ArrayOfTuples |> scatter(plt3, _, c=:firebrick1, markeralpha=0.3)
end


using CSV

list = CSV.File("/storage/home/suv87/work/julia/grdpg/code/datasets/email-Eu-core-department-labels.txt", delim=" ")

list = list |> Tables.matrix
list[idx, 2]
