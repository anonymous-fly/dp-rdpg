n = 100

# SBM
p, q, r = 0.4, 0.1, 0.15
f = (x, y) -> r + p * (x == y) - q * (x != y)
Z = rand([1, 2, 3], n);
A = Adjacency(f, Z);

# Sociability Network
Z = rand(Gamma(1, 1), n)
f = (x, y) -> 1 - exp(-2 * x * y)
A = Adjacency(f, Z);

# Circle
σ = 0.5
d = 1
M = Sphere(d)
Z = rdpg.randSphere(n, d = d)
f = (x, y) -> min(1, pdf(Normal(0, σ), distance(M, x, y)))
A = Adjacency(f, Z);


# Lemniscate
σ = 0.75
Z = randLemniscate(n)
f = (x, y) -> min(1, pdf(Normal(0, σ), lemniscate_distance(x, y)))
A = Adjacency(f, Z);