_Matrix_to_ArrayOfTuples = M -> tuple.(eachcol(M)...)
_ArrayOfTuples_to_Matrix = A -> hcat(collect.(A)...)'
_ArrayOfVectors_to_ArrayOfTuples = A -> Tuple.(A)
_ArrayOfTuples_to_ArrayOfVectors = A -> [[a...] for a in A]

m2t = _Matrix_to_ArrayOfTuples
t2m = _ArrayOfTuples_to_Matrix
v2t = _ArrayOfVectors_to_ArrayOfTuples
t2v = _ArrayOfTuples_to_ArrayOfVectors

function read_network(filename)
    return LightGraphs.LinAlg.symmetrize(adjacency_matrix(
        loadgraph(filename, "network", EdgeListFormat())
    ))
end;

function V(V, fun = mean; slice = 1, i = 1)
    if slice == 1
        reshape(fun(V, dims = 1), size(V, 3), size(V, 2), :)[i, :]
    elseif slice == 2
        reshape(fun(V, dims = 1), size(V, 3), size(V, 2), :)[:, i]
    end
end