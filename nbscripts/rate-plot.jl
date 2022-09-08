using DrWatson
@quickactivate projectdir()

using Plots

rate_1(c) = n -> log(n)^c / √n
rate_2 = n -> (log(n) / n)^(1 / 8)
# epsilon_rate = n -> log(1 + sqrt(log(log(n)) / log(n)))
epsilon_rate = n -> log(1 + log(log(n)) / log(n))
epsilon_transform(f) = n -> sqrt((exp(f(n)) + 1) / (exp(f(n)) - 1))


f(c) = n -> (epsilon_transform(epsilon_rate)(n) * rate_1(c)(n)) + rate_2(n)
g(c) = n -> (epsilon_transform(epsilon_rate)(n)^(5 / 2) * rate_1(c)(n)^(1 / 2)) + (epsilon_transform(epsilon_rate)(n)^(7 / 2) * rate_2(n))

theme(:default)
C = [1.2, 1.5]
cls1 = [:firebrick1, :red]
cls2 = [:black, :blue]
lty = [:solid, :dash]
plt = plot(0, 0, ylab="error", xlab="n", xscale=:log10)
for (c, i) in zip(C, eachindex(C))
    if i != 1
        plt = plot(plt, n -> 1, 10, 1e5, la=0, label=" ", xticks=10 .^ [0:8...])
    end
    plt = plot(plt, n -> f(c)(n), 1, 1e5, lw=3, la=0.5, ls=lty[i], c=cls1[i], label="𝔤(ϵ, n), c=$c")
    plt = plot(plt, n -> g(c)(n), 1, 1e5, lw=3, la=0.5, ls=lty[i], c=cls2[i], label="𝔣(ϵ, n), c=$c")

end
plt = plot(plt, legend=:topright)

# savefig(plot(plt, size=(600, 350)), plotsdir("rate.svg"))