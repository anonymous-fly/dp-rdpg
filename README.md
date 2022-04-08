# DP RDPG

Topological inference from differentially private random dot-product graphs.

## Getting Started

To get started, first clone the repository and start Julia.

```bash
$ git clone https://gitlab.com/anonymous-fly/dp-rdpg.git
$ cd ./dp-rdpg
$ julia
```

From the Julia REPL, you can enter the package manager by typing `]`, and activate the project environment with the required packages from the `Project.toml` file as follows.
```julia
julia>]
pkg> activate .
pkg> instantiate .
```

## Contents

The [notebooks](./notebooks/) directory contains the Jupyter notebooks for the experiments and simulations. The directory contains the following files:

- [x] [`example.ipynb`](./notebooks/example.ipynb): An illustration of persistence diagrams from spectral embeddings under $`\epsilon`$-differential privacy.


- [ ] [`clustering.ipynb`](./notebooks/clustering.ipynb): Topology-aware spectral clustering with more details. 


- [ ] [`sbm.ipynb`](./notebooks/sbm.ipynb): Illustration of persistence diagrams for community detection in stochastic blockmodels.


- [ ] [`simulation.ipynb`](./notebooks/simulation.ipynb): Simulations comparing convergence in bottleneck distance for $`\epsilon`$ known vs. $`\epsilon`$ unknown. 


- [ ] [`comparison.ipynb`](./notebooks/comparison.ipynb): Simulations comparing edgeFlip with LaplaceFlip. 


- [ ] [`real-world.ipynb`](./notebooks/real-world.ipynb): Illustration of the benefit of the topological perspective on real-world data. 


The [code](./code/) directory contains the `.jl` source-code for the analyses.


## Troubleshooting

If you want to view the Jupyter notebooks in the browser itself, please make sure you select the option to *"display rendered file"* to render the file in the browser, and click on *"load it anyway"* to display the file (most files are larger than the default 1.00MiB limit on Gitlab).
