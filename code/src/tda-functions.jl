_log_transform_interval(x, minval) = PersistenceInterval( tuple(max.(log.(x), fill(minval, 2))...) )

function log_transform_diagram(D)
    logD = deepcopy(D)
    minval = log(maximum(D[1][1]))
    for i in 1:length(logD)
        logD[i] = PersistenceDiagram(_log_transform_interval.(logD[i], minval), dim=i-1)
    end
    return logD
end

function diagram(X; d=2, log_trans=false, alpha=true)

    points = tuple.(eachcol(X)...)
    
    if alpha
        dgm = ripserer(Alpha(points), dim_max=d)
    else
        dgm = ripserer(EdgeCollapsedRips(points), dim_max=d)
    end

    return log_trans ? log_transform_diagram(dgm) : dgm
end


function randLemniscate(n; s = 0)

    signal = R"tdaunif::sample_lemniscate_gerono($n)"
    noise = R"matrix(rnorm(2 * $n, 1, $s), ncol=2)"

    X = tuple.(eachcol(rcopy(signal + noise))...)

    return X
end



function randCircle(n; s = 0)

    signal = R"TDA::circleUnif($n)"
    noise = R"matrix(rnorm(2 * $n, 1, $s), ncol=2)"
    X = tuple.(eachcol(rcopy(signal + noise))...)

    return X
end

function randSphere(n; d = 2)
    S = Sphere(d)
    X = [tuple(random_point(S)...) for i in 1:n]
    return X
end