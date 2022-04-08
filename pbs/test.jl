using Plots
println(ENV["GKSwstype"])
plt = plot(0:0.1:2, x-> log(x))
plt = plot(plt, 0:0.1:2, x-> sqrt(x))
savefig(plt, "./plots/plot.svg")
savefig(plt, "./plots/plot.pdf")
savefig(plt, "./plots/plot.png")
savefig(plt, "./plots/plot.html")
