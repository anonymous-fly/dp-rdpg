using DrWatson
@quickactivate "projectdir()"

include(srcdir("rdpg.jl"))
import Main.rdpg
using PersistenceDiagrams, Pipe, Plots, ProgressMeter, Random, Ripserer, Statistics

begin
    function scale_embeddings(X)
        return (X .- mean(eachrow(X))') * (X'X)^(-0.5)
    end

    function diagram(X, dim_max; alpha=true)
        points = tuple.(eachcol(X)...)
        dgm = ripserer(Alpha(points), dim_max=dim_max)
        return dgm
    end

    function bottleneck_distances(X, Y, dim_max)
        DX = diagram(X, dim_max)
        DY = diagram(Y, dim_max)
        return [Bottleneck()(DX[d], DY[d]) for d in 1:dim_max+1]
    end

    function generate_sbm(n, k, p, r)
        f = (x, y) -> (r + p * (x == y))   * log(n) / n
        Z = rand(1:k, n)
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
            X = rdpg.scale_embeddings(X)
            X_private = rdpg.scale_embeddings(X_private)
        end
        return bottleneck_distances(X, X_private, d)
    end
end


begin
    repeats = 20
    Ks = [1 / 2, 2 / 3, 9 / 10]
    # Ks_legend = ["1/2", "2/3", "1"]
    Ks_legend = ["0.50", "0.66", "0.90"]
    N = [30, 50, 100, 200, 400, 600, 800, 1000]
end

begin
    p, r = 6, 1, 0.15
    clust = 3
    n = length(N)
end


begin
    results_3 = [zeros(repeats, n) for _ in 1:length(Ks)]
    prog = Progress(convert(Int, n * repeats * length(Ks)))
    for i in 1:n
        A = generate_sbm(N[i], clust, p, r)
        for j in 1:repeats
            for k in 1:length(Ks)

                ϵn = log(N[i])^(Ks[k])
                error = simulate_one(A, 0, ϵn, :eps)
                results_3[k][j, i] = error[1]
                next!(prog)

            end
        end
    end
end

begin
    plt = plot(title="ϵ=logᵏ(n)", xlabel="n", ylabel="Bottleneck")
    for k in 2:length(Ks)
        plot!(plt, N,
            mean(results_3[k], dims=1)',
            # ribbon=std(results_3[k], dims=1),
            markershape=:o,
            label="k=$(Ks_legend[k])",
            lw=2, fillapha=0.01,
        )
    end

    plt = plot(plt, size=(300,200))
    
    # savefig(plotsdir("sbm/convergence4.svg"))
    # savefig(plotsdir("sbm/convergence4.pdf"))
end

