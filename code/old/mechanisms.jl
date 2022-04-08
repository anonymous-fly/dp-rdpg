# Mechanisms

# 1. Symmetric Edge Flip
_flipSingleEdge(x, p) = rand(Bernoulli(p)) ? 1 - x : x


function edgeFlip(A; ϵ::Real = 1, p::Real = -1, parameters=nothing)
    if p < 0
        p = privacy(ϵ = ϵ)
    end

    n = size(A, 1)
    X = sparse([i>j ? _flipSingleEdge(A[i, j], p) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end;



# 2. Laplace Edge Flip

_LapFlipEdge(x, ϵ) = x + rand(Laplace(0, 1/ϵ)) > 0.5 ? 1 : 0

function laplaceFlip(A; ϵ::Real = 1, parameters=nothing)

    n = size(A, 1)
    X = sparse([i>j ? _LapFlipEdge(A[i, j], ϵ) : 0 for i = 1:n, j = 1:n])

    return LightGraphs.LinAlg.symmetrize(X)
end;



# 3. Select-Measure-Reconstruct-type mechanism

_extractEdges(A) = [A[i, j] for i in 1:size(A, 1), j in 1:size(A, 1) if j>i]

function SMR_Mechanism(A; ϵ::Real = 1, parameters=(0.5, 0.5, 100))
    p, q, K  = parameters
    ϵ1 = p * ϵ
    ϵ2 = q * ϵ / K
    n = size(A, 1)

    x = _extractEdges(A)

    M = sum(x) + rand(Laplace(0, ϵ1))

    Σ = []
    for i in 1:K
        push!(Σ, _extractEdges(edgeFlip(A; ϵ=ϵ2)))
    end

    x̂ = SMR_Reconstruct(Σ, M, K)

    Â = spzeros(n,n)

    for i in 1:(n-1)
        for j in (i+1):n
            k = convert(Int, (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
            Â[i, j] = x̂[k]
        end
    end
    return LightGraphs.LinAlg.symmetrize(Â)
end


function SMR_Reconstruct(Σ, M, K)
    model = Model(Gurobi.Optimizer);
    n = size(Σ[1],1)
    @variable(model, z[1:n], Bin)
    @constraint(model, M-1 ≤ sum(z) ≤ M+1)
    # @constraint(model, sum(z) > 0)
    obj = 0
    for i in 1:K
        obj = obj + sum(t -> t^2, z - Σ[i])
    end

    @objective(model, Min, obj)
    optimize!(model)
    return value.(z) .> 0
end



# 4.

_incidence_matrix(A; sign=true) = LightGraphs.LinAlg.incidence_matrix(SimpleGraph(A); oriented=sign)


_laplacian_matrix(A) = LightGraphs.LinAlg.laplacian_matrix(SimpleGraph(A))


function full_incidence_matrix(A; sign=true)
    n_v = size(A, 1)
    n_e = binomial(n_v, 2)

    B = spzeros(n_e, n_v)

    k=1

    for i in 1:(n_v-1)
        for j in (i+1):n_v
            if A[i, j] != 0
                B[k, i] = (sign ? -1 : 1) * A[i, j]
                B[k, j] = A[i, j]
                k+=1
            end
        end
    end

    return B
end



################################################


# Trying to make a differentially private version of https://arxiv.org/pdf/1410.4273.pdf


_F(t, n, m, l) = (1 - (n / t)) * ((l / (m + t - n - (0.5 * (l-1)))) - n/t)

function get_T(n, m, l)
    term1 = n * (m - n + (0.5 * (l+1)))
    term2 = l * (m - (0.5 * (l-1))) * term1
    term3 = l - n
    t̂ = ( term1 + √(term2) / term3 )
    return t̂ * (1 + _F(t̂, n, m, l))
end



function sparsify(A; l=0)

    B = full_incidence_matrix(A)

    if l < 1
        l = Int(round(size(B,1) / 2))
    end

    Φ = svds(B)[1]

    Π = []
    iter = 1

    while iter <= l
        (U, σ, V) = (Φ.U', Φ.S, Φ.Vt);
        n, m = size(U)
        T = get_T(n, m, l)
        C = zeros(Int, n, n)

        SearchSet = setdiff(collect(1:m), Π)

        Λ, Ψ = eigs(C; nev=n-1);
        if length(Λ) < n
            Λ = vcat(Λ,zeros(n-length(Λ)))
        end

        λmin, _ = eigs(A; nev=1, which=:SR);

        # Step 1
        @var μ
        F = System([tr(inv(C - (1/μ) * I(n))) - T], variables=[μ])
        result  = HomotopyContinuation.solve(F)

        λ = map(x -> 1/x[1][1] .< minimum(Λ) ? 1/x[1][1] : 0, vec(real_solutions(result)))[1]


        # G(x) = tr(inv(C - x * I(n))) - T

        # Step 2
        @var λ̂
        f₁ = λ̂ - λ
        f₂ = m - iter + sum([ (1 - Λ[i]) / (Λ[i] - λ) for i in 1:n])
        f₃ = sum([ (1 - Λ[i]) / ( (Λ[i] - λ) * (Λ[i] - λ̂) ) for i in 1:n])
        f₄ = sum([ 1 / ( (Λ[i] - λ) * (Λ[i] - λ̂) ) for i in 1:n])

        F2 = System([f₁ * f₂ - (f₃ / f₄)], variables=[λ̂])
        result2 = HomotopyContinuation.solve(F2)

        λ̂ = real_solutions(result2)[1][1]


        # Step3

        for i in SearchSet
            lhs = tr(inv(C - (λ̂ * I(n)) + (U[:,i] * U[:,i]')))
            rhs = tr(inv(C - (λ * I(n))))
            if lhs < rhs
                C = C + (U[:,i] * U[:,i]')
                Π = union(Π, i)
                break
            end
        end

        iter += 1

    end

    X = sparse(zeros(Int8, n, n))

    for k in Π
        (i, j) = findall(!=(0), B[k,:])
        X[i, j] = 1
        X[j, i] = 1
    end

    return X
end


function SparseMechanism(A; ϵ::Real = 1, parameters=(0.5, 0.5))
    p, q  = parameters
    ϵ1 = p * ϵ
    ϵ2 = q * ϵ
    n = size(A, 1)

    M = sum(A)/2 + rand(Laplace(0, ϵ1))
    Y = edgeFlip(A; ϵ=ϵ2)

    L = (M + (sum(Y) / 2)) / 2
    # L = Int(round(L))
    L = Int(round(M))
    X = sparsify(Y; l=M)

    return X
end



# Low Rank Reconstruction Mechanism

function LowRankReconstructMechanism(A; ϵ::Real = 1, dims = 3)
    # TODO: rename `parameters` to `d` if kwargs change lands...
    # Done
    # d = parameters
    λ, v = eigs(A, nev = dims, maxiter=500)

    P = clamp.(v * diagm(λ) * v' , 1 / (exp(ϵ) + 1), exp(ϵ) / (exp(ϵ) + 1))

    return Adjacency(identity, P)

end


## Johnson-Lindenstrauss
function JLMechanism(A; ϵ::Real = 1, delta::Real = 0.001, η::Real = 0.05, ν::Real = 0.05)
    r = ceil(8 * log(2 / ν) / η^2)
    w = sqrt(32 * r * log(2 / delta)) * log(4*r / delta) / ϵ
    
    n = size(A, 1)
    M = randn((Int(r), binomial(n, 2)))

    # incidence matrix after transforming adjacency matrix
    A = Array(A) .* (1 - w/n) .+ w/n
    B = full_incidence_matrix(A, sign=false)

    L = B' * M' * M * B ./ r

    A = sparse(replace(x -> x < -0.5 ? 1 : 0, L))
    A[diagind(A)] .= 0

    return A
end

