function graphplt(A; Z = nothing, α = 0.1, scheme = :viridis, color = "red")
    if Z !== nothing
        zcolors = cgrad(scheme, length(unique(Z)), rev = true, categorical = true)[Z]
        gplot(SimpleGraph(A), nodefillc = zcolors, edgestrokec = coloralpha(colorant"grey", α))
    else
        gplot(SimpleGraph(A), nodefillc = parse(Colorant, color), edgestrokec = coloralpha(colorant"grey", α))
    end
end;


function spectralplt(X; Z = nothing, α = 1, scheme = :viridis, color = "red", lim=(-3,3))
    if Z !== nothing
        zcolors = cgrad(scheme, length(unique(Z)), rev = true, categorical = true)[Z]
        plt = plot(X[:, 1], X[:, 2], X[:, 3], seriestype = :scatter, label = false, c = zcolors, ratio=1, lim=lim)
        display(plt)
    else
        plt = plot(X[:, 1], X[:, 2], X[:, 3], seriestype = :scatter, label = false, c = parse(Colorant, color), ratio=1, lim=lim)
        display(plt)
    end
        
    return plt
end;