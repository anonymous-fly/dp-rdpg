# DP RDPG

Topological inference from differentially private random dot-product graphs.

## Getting Started

To get started, first clone the repository and start Julia.

```bash
$ git clone https://github.com/anonymous-fly/dp-rdpg.git
$ cd ./dp-rdpg
$ julia
```

From the Julia REPL, you can enter the package manager by typing `]`, and activate the project environment with the required packages from the `Project.toml` file as follows.
```julia
julia> ]
pkg> activate .
pkg> instantiate .
```

## Contents

The [notebooks](./notebooks/) directory contains the Jupyter notebooks for the experiments and simulations. The directory contains the following files:

- [x] [`example.ipynb`](./notebooks/example.ipynb): An illustration of persistence diagrams from spectral embeddings under <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-differential privacy.


- [ ] [`clustering.ipynb`](./notebooks/clustering.ipynb): Topology-aware spectral clustering with more details. 


- [ ] [`sbm.ipynb`](./notebooks/sbm.ipynb): Illustration of persistence diagrams for community detection in stochastic blockmodels, including the relationship between <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> and <img src="https://render.githubusercontent.com/render/math?math=n"> vis-à-vis [Seif, et. al, (2022)](https://arxiv.org/abs/2202.00636).


- [ ] [`simulation.ipynb`](./notebooks/simulation.ipynb): Simulations comparing convergence in bottleneck distance for <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> known vs. <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> unknown. 


- [ ] [`comparison.ipynb`](./notebooks/comparison.ipynb): Simulations comparing edgeFlip with LaplaceFlip. 


- [ ] [`real-world.ipynb`](./notebooks/real-world.ipynb): Illustration of the benefit of the topological perspective on real-world data. 


The [code](./code/) directory contains the `.jl` source-code for the analyses.


## Troubleshooting

The code here uses the [Ripserer.jl](https://github.com/mtsch/Ripserer.jl) backend for computing persistent homology. The exact computation of persistent homology is achieved using the **Alpha** complex which, additionally, uses the ![MiniQHull.jl](https://github.com/gridap/MiniQhull.jl) library, which has a known incompatibility with the Windows operating system (![see here](https://github.com/gridap/MiniQhull.jl/issues/5)). If you're using Windows, then you can either:
1. Use the windows subsystem for linux to run the code here, or
2. You can change the relevant parts of the code to not use the Alpha complex, e.g., `Alpha(Xn)` => `Xn`. 

For any other issues, please click [here](https://github.com/anonymous-fly/dp-rdpg/issues/new/choose).