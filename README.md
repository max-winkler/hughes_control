# Hughes control: A software for computing optimal evacuation of a crowd

## Prerequisites

The software is implemented in `Python`. The software relies on the  modules `numpy`, `scipy`, `FEniCs=2019.1`, `mshr` and `superlu_dist=6` that are needed to execute the program. These modules can easily be installed in a `conda` environment (see [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)):

```bash
conda create -n myenv python=3.8
conda activate myenv
conda install -c conda-forge numpy scipy fenics=2019.1 mshr=2019.1 superlu_dist=6
```
Resolving the dependencies in the latter step took 15 minutes on my machine. So be patient.

## Setting up the problem

Before starting the computation you probably want to set the model and algorithm parameters, want to define a computational domain and boundary conditions. To do so you have to modify the source files as follows:

All the regularization parameters have to be defined at the beginning of the file `hughes_solution.py`. Instead, all model related data are defined in an example class in the file `setup_example.py`. To implement your own scenario copy one of the example classes and adapt the parameters. The following variables have to be provided by each instance:

- `T`: the final time,
- `N`: The number of time steps for the temporal discretization,
- `mesh`:  defines the computational domain,
- `exits`: an instance of a classes implementing a function `inside(x, on_boundary)` returning a True if the point `x` is on the outflow boundary,
- `walls`: an instance of a classes implementing a function `inside(x, on_boundary)` returning a True if the point `x` is on a wall boundary,
- `rho_0`: the initial density as UFL Expression,
- `ag_pos_0`: a numpy matrix of dimension $n_{agents}\times 2$ storing the initial positions of the agents,
- `ag_vel_0`: a numpy matrix of dimension $n_{agents}\times 2$ storing the initial agent velocity. This defines the initial control `u` which is constant in time,
- `wall_region(x)`: a function returning a boolean used to mark wall boundary parts,
- `inside_room(x)`: a function returning a boolean which is True is the point `x` is in the subregion where densities are penalized in the objective.

## Running the program

Run the program with:

```bash
python3 hughes_control.py
```
