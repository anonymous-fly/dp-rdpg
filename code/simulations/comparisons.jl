import Pkg;
Pkg.activate(".");

include("../src/rdpg.jl")
import Main.rdpg
using Distributions, Pipe, ProgressBars


begin
    n = 500
    d = 1
    dim = 3
    ϵ = 1
    M = Sphere(d)

    Z = rdpg.randSphere(n, d = d)
    # Z = [0.5 .* (z .+ (0.5, 0.5)) for z in Z]
    dist_max = pairwise(Distances.Euclidean(), Z) |> maximum
    f = (x, y) -> distance(Manifolds.Euclidean(), x, y) / dist_max

    A = rdpg.Adjacency(f, Z)
    X, _ = rdpg.spectralEmbed(A, d = 3, scale = false)
    plt1 = @pipe X |>
                 rdpg._Matrix_to_ArrayOfTuples |>
                 scatter(_, c = :dodgerblue, ratio = 1, lim = (-2, 2))

    B = @pipe A |> rdpg.edgeFlip(_, ϵ = ϵ)
    Y1, _ = rdpg.spectralEmbed(B, d = 3, scale = false)
    plt2 = @pipe Y1 |> rdpg._Matrix_to_ArrayOfTuples |>
                 scatter(_, c = :firebrick, ratio = 1, lim = (-2, 2))


    C = (B .- (rdpg.τ(ϵ)^2)) ./ (rdpg.σ(ϵ)^2)
    Y2, _ = rdpg.spectralEmbed(C, d = 3, scale = false)
    plt3 = @pipe Y2 |> rdpg._Matrix_to_ArrayOfTuples |>
                 scatter(_, c = :green, ratio = 1, lim = (-2, 2))

    Y3 = Y1 ./ rdpg.σ(ϵ)
    plot(plt1, plt2, plt3, size = (1200, 400), layout = (1, 3))
end



function oneSim(; n, ϵ, dim = 3, scale = false)
    begin
        d = 1
        ϵ = 1

        Z = rdpg.randSphere(n, d = d)
        dist_max = pairwise(euclidean, Z) |> maximum
        f = (x, y) -> distance(Euclidean(), x, y) / dist_max

        A = rdpg.Adjacency(f, Z)
        X, _ = rdpg.spectralEmbed(A, d = 3, scale = false)

        B = @pipe A |> rdpg.edgeFlip(_, ϵ = ϵ)
        Y1, _ = rdpg.spectralEmbed(B, d = 3, scale = false)

        C = (B .- (rdpg.τ(ϵ)^2)) ./ (rdpg.σ(ϵ)^2)
        Y2, _ = rdpg.spectralEmbed(C, d = 3, scale = false)

        Y3 = Y1 ./ rdpg.σ(ϵ)
    end

    if scale
        D = rdpg.diagram(X |> rdpg.scale_embeddings)
        D1 = rdpg.diagram(Y1 |> rdpg.scale_embeddings)
        D2 = rdpg.diagram(Y2 |> rdpg.scale_embeddings)
        D3 = rdpg.diagram(Y3 |> rdpg.scale_embeddings)
    else
        D = rdpg.diagram(X)
        D1 = rdpg.diagram(Y1)
        D2 = rdpg.diagram(Y2)
        D3 = rdpg.diagram(Y3)
    end

    return (
        bottleneck = [Bottleneck()(D, D1), Bottleneck()(D, D2), Bottleneck()(D, D3)],
        points = [X, Y1, Y2, Y3]
    )

end


N = [50, 100, 200, 400]
ϵ = [0.5, 1, 2, 4, 10]

n = length(N)
m = length(ϵ)
repeats = 20

neps = zeros(repeats, n, m)
weps = zeros(repeats, n, m)
peps = zeros(repeats, n, m)


for i in tqdm(1:n)
    for j in 1:m
        for k in 1:repeats
            res = oneSim(n = N[i], ϵ = ϵ[j], scale=true)
            neps[k, i, j] = res.bottleneck[1]
            weps[k, i, j] = res.bottleneck[2]
            peps[k, i, j] = res.bottleneck[3]
        end
        println("Iteration $((i,j)). No eps: $(neps[:, i, j] |> mean), With eps: $(weps[:, i, j] |> mean), Post-process: $(peps[:, i, j] |> mean)")
    end
end


plt1 = plot(title = "ϵ publicly available")
i = 1;
plot!(plt1, N, rdpg.V(weps, mean; slice = 1, i = i), ribbon = rdpg.V(weps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])");
i = 2;
plot!(plt1, N, rdpg.V(weps, mean; slice = 1, i = i), ribbon = rdpg.V(weps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])");
i = 3;
plot!(plt1, N, rdpg.V(weps, mean; slice = 1, i = i), ribbon = rdpg.V(weps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])", ylim = (0, 1));
plt1

plt2 = plot(title = "ϵ not publicly available")
i = 1;
plot!(plt2, N, rdpg.V(neps, mean; slice = 1, i = i), ribbon = rdpg.V(neps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])");
i = 2;
plot!(plt2, N, rdpg.V(neps, mean; slice = 1, i = i), ribbon = rdpg.V(neps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])");
i = 3;
plot!(plt2, N, rdpg.V(neps, mean; slice = 1, i = i), ribbon = rdpg.V(neps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])", ylim = (0, 1));
plt2


plt3 = plot(title = "ϵ Post Scaled")
i = 1;
plot!(plt3, N, rdpg.V(peps, mean; slice = 1, i = i), ribbon = rdpg.V(peps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])");
i = 2;
plot!(plt3, N, rdpg.V(peps, mean; slice = 1, i = i), ribbon = rdpg.V(peps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])");
i = 3;
plot!(plt3, N, rdpg.V(peps, mean; slice = 1, i = i), ribbon = rdpg.V(peps, std; slice = 1, i = i), label = "ϵ = $(ϵ[i])", ylim = (0, 1));
plt2

plot(plt1, plt2, plt3, size=(1200, 350), layout=(1,3))


#TODO:
# Rescale X, Y1 and Y2. Compare the results now. They're all on the same scale. 
