# 3DGS

This project will implement (based on [this paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)) 3D Gaussian Splatting in Vulkan and CUDA.

Naturally, Vulkan will be used to visualise generated/approximated gaussians, while CUDA will be used for data preprocessing, and gradient descent.

## First step

First goal is to create a CPU gaussian renderer that will:
- Generate result in `PPM` format
- Visualise 3D Gaussians (as projected on 2D screen)
- Try to implement simple color blending

## Second step

Accelerate computations from `First step` using GPU (Vulkan)

TODO

## Third step

CUDA accelerated gaussian parameters updates (using Gradient descent)
