results_dense = [zeros(repeats, n) for _ in 1:length(Ks)];
results_sparse = [zeros(repeats, n) for _ in 1:length(Ks)];

prog = Progress(convert(Int, n * repeats * length(Ks)))

Random.seed!(2022)
for i in 1:n
    for j in 1:repeats
        A_sparse = generate_sbm_sparse(N[i], 3, 9, 1)
        A_dense = generate_sbm_dense(N[i], 3, 0.7, 0.05)
        for k in 1:length(Ks)

            ϵn = 5 * log(N[i])^(Ks[k])

            error_sparse = simulate_one(A_sparse, 0, ϵn, :eps)
            error_dense = simulate_one(A_dense, 0, ϵn, :eps)

            results_sparse[k][j, i] = error_sparse[1]
            results_dense[k][j, i] = error_dense[1]

            next!(prog)

        end
    end
end

theme(:default)
plt_sparse = plot(title="ϵ=logᵏ(n)", xlabel="n", ylabel="Bottleneck Distance")
for k in 1:length(Ks)
    plot!(plt_sparse, N,
        mean(results_sparse[k], dims=1)',
        # ribbon=std(results_3[k], dims=1),
        marker=:o,
        label="k=$(Ks_legend[k])",
        lw=3, fillapha=0.01,
    )
end


plt_dense = plot(title="ϵ=logᵏ(n)", xlabel="n", ylabel="Bottleneck Distance")
for k in 1:length(Ks)
    plot!(plt_dense, N,
        mean(results_dense[k], dims=1)',
        # ribbon=std(results_dense[k], dims=1),
        marker=:o,
        label="k=$(Ks_legend[k])",
        lw=3, fillapha=0.01,
    )
end


savefig(plt_dense, plotsdir("temp/plot_dense.svg"))
savefig(plt_sparse, plotsdir("temp/plot_sparse.svg"))
