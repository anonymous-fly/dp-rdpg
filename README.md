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


- [x] [`clustering.ipynb`](./notebooks/clustering.ipynb): Topology-aware spectral clustering with more details. 


- [x] [`sbm.ipynb`](./notebooks/sbm.ipynb): Illustration of persistence diagrams for community detection in stochastic blockmodels, including the relationship between <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> and <img src="https://render.githubusercontent.com/render/math?math=n"> vis-à-vis [Seif, et. al, (2022)](https://arxiv.org/abs/2202.00636).


- [x] [`simulation.ipynb`](./notebooks/simulation.ipynb): Simulations comparing convergence in bottleneck distance for <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> known vs. <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> unknown. 


- [x] [`comparison.ipynb`](./notebooks/comparison.ipynb): Simulations comparing edgeFlip with LaplaceFlip. 


- [x] [`EuCore.ipynb`](./notebooks/eucore.ipynb): Illustration of the edgeFlip & persistent homology using the ![email-Eu-Core](http://www.cs.cmu.edu/~jure/pubs/powergrowth-tkdd.pdf) dataset, comprising of email communications between researchers from a large European research institution.


The [code](./code/) directory contains the `.jl` source-code for the analyses. All functions prefixed with `rdpg.` are defined in the [src](./src/) directory, and is imported through the [`rdpg` module](./src/rdpg.jl).


## Troubleshooting

The code here uses the [Ripserer.jl](https://github.com/mtsch/Ripserer.jl) backend for computing persistent homology. The exact computation of persistent homology is achieved using the **Alpha** complex which, additionally, uses the ![MiniQHull.jl](https://github.com/gridap/MiniQhull.jl) library, which has a known incompatibility with the Windows operating system (![see here](https://github.com/gridap/MiniQhull.jl/issues/5)). If you're using Windows, then you can either:
1. Use the windows subsystem for linux to run the code here, or
2. You can change the relevant parts of the code to not use the Alpha complex, e.g., `Alpha(Xn)` => `Xn`. 

For any other issues, please click [here](https://github.com/anonymous-fly/dp-rdpg/issues/new/choose).