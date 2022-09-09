using DrWatson
@quickactivate projectdir()

using Plots

rho(n) = log(n) / n
rate_1(c) = n -> log(n)^c / √(n * rho(n))
rate_2 = n -> (log(n) / n)^(1 / 8)
# epsilon_rate = n -> log(1 + sqrt(log(log(n)) / log(n)))
epsilon_rate1 = n -> log(log(n))
epsilon_rate2 = n -> n# - log(log(n))
epsilon_transform(f) = n -> sqrt((exp(f(n)) + 1) / (exp(f(n)) - 1))


h(c) = n -> (epsilon_transform(epsilon_rate1)(n) * rate_1(c)(n)) + rate_2(n)
g(c) = n -> (epsilon_transform(epsilon_rate2)(n) * rate_1(c)(n)) + rate_2(n)


begin
    
    theme(:default)
    C = [2]
    cls1 = [:firebrick1, :red]
    cls2 = [:black, :blue]
    lty = [:solid, :dash]
    plt = plot(0, 0, ylab="error", xlab="n", xscale=:log10)
    for (c, i) in zip(C, eachindex(C))
        if i != 1
            plt = plot(plt, n -> 1, 10, 1e10, la=0, label=" ", xticks=10 .^ [0:8...])
        end
        plt = plot(plt, n -> h(c)(n), 10, 1e10, lw=3, la=0.5, ls=lty[i], c=cls1[i], label="𝔤(ϵ, n), c=$c")
        plt = plot(plt, n -> g(c)(n), 10, 1e10, lw=3, la=0.5, ls=lty[i], c=cls2[i], label="𝔣(ϵ, n), c=$c")
        
    end
    plt = plot(plt, legend=:topright)

end
# savefig(plot(plt, size=(600, 350)), plotsdir("rate.svg"))