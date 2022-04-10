using DrWatson
@quickactivate "projectdir()"

includet(srcdir("rdpg.jl"))
import Main.rdpg
using PersistenceDiagrams, Pipe, Plots, ProgressMeter, Random, Ripserer, Statistics, StatsBase
using Distances, Manifolds

begin

    function scale_embeddings(X)
        return (X .- mean(eachrow(X))') * (X'X)^(-0.5)
    end

    function diagram(X, dim_max; alpha=true)
        points = tuple.(eachcol(X)...)
        # dgm = ripserer(Alpha(points), dim_max=dim_max)
        dgm = ripserer(points, dim_max=1)
        return dgm
    end

    function bottleneck_distances(X, Y, dim_max)
        DX = diagram(X, dim_max)
        DY = diagram(Y, dim_max)
        return [Bottleneck()(DX[d], DY[d]) for d in 2:2]
    end

function generate_graph(n)
    Z = rdpg.randSphere(n, d=1)
    dist_max = pairwise(Distances.Euclidean(), Z) |> maximum
    f = (x, y) -> distance(Manifolds.Euclidean(), x, y) / dist_max
    return rdpg.Adjacency(f, Z)
end

    function simulate_one(A, d, epsilon, method)
        X, _, _ = rdpg.spectralEmbed(A, d=d + 1, scale=false)
        A_private = rdpg.edgeFlip(A, ϵ=epsilon)

        if method == :eps
            A_private = A_private .- rdpg.privacy(ϵ=epsilon)
        end

        X_private, _, _ = rdpg.spectralEmbed(A_private, d=d + 1, scale=false)

        if method == :eps
            X_private = X_private ./ (1 - 2 * rdpg.privacy(ϵ=epsilon))
        elseif method == :noeps
            X = StatsBase.standardize(ZScoreTransform, X, dims=1)
            X_private = StatsBase.standardize(ZScoreTransform, X_private, dims=1)
        end
        return bottleneck_distances(X, X_private, 2)
    end

end;



begin

    repeats = 5
    Eps = [0.50, 1.00, 2.00, 4.00]
    Eps_legend = ["0.50", "1.00", "2.00", "4.00"]
    N = [20, 40, 60, 80, 100, 200, 400, 600]

    p, r = 0.4, 0.1, 0.15
    clust = 3
    n = length(N)

end


begin

    repeats = 20
    Eps = [2 / 3, 9 / 10]
    Eps_legend = ["0.8", "1.0"]

    # Eps = [1.00, 2.00, 4.00, 10.0]
    # Eps_legend = ["0.50", "1.00", "2.00", "4.00"]

    N = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    # N = [20, 40, 60, 80, 100, 200, 400, 600]

    n = length(N)
    m = length(Eps)

end

begin

    results_w_eps = [zeros(repeats, n) for _ in 1:length(Eps)]
    results_no_eps = [zeros(repeats, n) for _ in 1:length(Eps)]

    prog = Progress(convert(Int, n * repeats * length(Eps)))

    for i in 1:n
        for j in 1:repeats
            A = generate_graph(N[i])
            for k in 1:length(Eps)

                ϵn = Eps[k]
                # ϵn = log(N[i])^(Eps[k])

                error_w_eps = simulate_one(A, 2, ϵn, :eps)
                error_no_eps = simulate_one(A, 2, ϵn, :noeps)

                results_w_eps[k][j, i] = error_w_eps[1]
                results_no_eps[k][j, i] = error_no_eps[1]

                next!(prog)

            end
        end
    end

end


begin
    plt1 = plot(title="ϵ publicly available", xlabel="n", ylabel="Bottleneck Distance")
    for k in 1:length(Eps)
        plot!(plt1, N,
            mean(results_w_eps[k], dims=1)',
            # ribbon = std(results_w_eps[k], dims=1),
            # markershape=:o,
            label="k=$(Eps_legend[k])",
            lw=3, fillapha=0.01,
        )
    end

    savefig(plt1, plotsdir("sims/e.svg"))

    plt2 = plot(title="ϵ not publicly available", xlabel="n", ylabel="Bottleneck Distance")
    for k in 1:length(Eps)
        plot!(plt2, N,
            mean(results_no_eps[k], dims=1)',
            # ribbon = std(results_no_eps[k], dims=1),
            # markershape=:o,
            label="k=$(Eps_legend[k])",
            lw=3, fillapha=0.01,
        )
    end
    savefig(plt2, plotsdir("sims/ne.svg"))

end



plot(plt1, plt2, size=(1000, 300))

