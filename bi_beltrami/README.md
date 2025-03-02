# Python Package for Surface Slip Boundary Condition Generation

## 1. Introduction

This package provides tools to generate surface slip boundary conditions for various computational simulations. 
The package is implemented in Python and allows users to create custom surface slip configurations based on their specific needs. 
It is designed to be user-friendly and does not require compilation, making it easy to integrate into existing workflows.

## 2. References


1. D. J. Acheson. Elementary fluid dynamics. Oxford University Press, 1990.

2. M. Arroyo and A. DeSimone. Relaxation dynamics of fluid membranes. Physical Review E—Statistical,
Nonlinear, and Soft Matter Physics, 79(3):031915, 2009.
3. B. J. Gross and P. J. Atzberger. Hydrodynamic flows on curved surfaces: Spectral numerical methods
for radial manifold shapes. Journal of Computational Physics, 371:663–689, 2018.
4. B. J. Gross, N. Trask, P. Kuberry, and P. J. Atzberger. Meshfree methods on manifolds for hydrodynamic flows on curved surfaces: A Generalized Moving Least-Squares (GMLS) approach. J. Comput. Phys., 409:109340, 2020.
5. M. L. Henle, R. McGorty, A. B. Schofield, A. Dinsmore, and A. Levine. The effect of curvature and topology on membrane hydrodynamics. Europhysics Letters, 84(4):48001, 2008.
6. J. K. Sigurdsson and P. J. Atzberger. Hydrodynamic coupling of particle inclusions embedded in curved lipid bilayer membranes. Soft matter, 12(32):6685–6707, 2016.

## Usage Instructions

### 1. Prepare the Package

The codes are implemented in Python (version 3.x), and there is no need to compile the package to use it. However, before running the package, ensure that you have pre-installed the necessary dependencies, such as `cvxopt`. You can install this package using pip:

```bash
pip install cvxopt
```

### 2. Create Surface Slip Boundary Conditions

#### Generate Your Own Slip Boundary Condition

First, navigate to the directory `bi_beltrami/` and inspect the input file `test_input.txt`. This file contains the necessary parameters and configurations for generating the surface slip boundary conditions.

---

```
# model parameters
nu              2
gamma           3

# mesh-free parameters
k0              51
degree          2

pde_solver      extrinsic

slip_save_type  velocity_field
save_directory  slip_files/
save_potential  True
error_display   True

# Load rigid bodies configuration, provide *.vertex files
structure   vertex_files/shell_N_601_Rg_1_FM_cube2sphere.vertex
```

---
* `nu`: (float) the surface fluid viscosity.

* `gamma`: (float) the coupling coefficient.

* `k0`: (int) k-NN.

* `degree`: (int) maximum degree of polynomial.

* `pde_solver`: Options: `extrinsic`,`intrinsic` and `optimized`
* `slip_save_type`: Options: `velocity_field`,`velocity_field and potential_field`


To run the code and generate the surface slip file, use the following command:

```bash
python generate_surface_slip.py --input-file test_input.dat
```

### 3. Package Organization

- **slip_files**: This directory is where we store the generated surface slip files. These files contain the output of the boundary condition generation process and can be used in further simulations.

- **vertex_files**: This directory contains configuration files of input structures. These files define the geometry and properties of the structures for which surface slip boundary conditions are being generated.

