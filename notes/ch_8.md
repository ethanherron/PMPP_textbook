# 8 Stencil

## Chapter Outline
- 8.1 Background
- 8.2 Parallel stencil: a basic algorithm
- 8.3 Shared memory tiling for stencil sweep
- 8.4 Thread coarsening
- 8.5 Register tiling
- 8.6 Summary

## Introduction

Stencils are foundational to numerical methods for solving partial differential equations in fields such as:
- Fluid dynamics
- Heat conductance
- Combustion
- Weather forecasting
- Climate simulation
- Electromagnetics

Key characteristics of stencils:
- Process discretized physical quantities (mass, velocity, force, temperature, etc.)
- Approximate derivative values of functions based on function values within a range
- Similar to convolution - both calculate new values based on neighborhood elements
- Unlike convolution, stencils:
  - Iteratively solve values of continuous, differentiable functions
  - May have dependencies requiring specific ordering constraints
  - Typically use high-precision floating point data

## 8.1 Background

### Discretization
- Converting continuous functions into discrete representations
- Example: Sine function discretized into grid points
- Representation quality depends on:
  - Grid spacing (smaller spacing → better accuracy but more storage/computation)
  - Numerical precision (double vs. single vs. half precision)

### Stencil Definition
- A geometric pattern of weights applied at each point of a structured grid
- Specifies how values at a point are derived from neighboring points
- Used for numerical approximation of derivatives in finite difference methods

### Finite Difference Approximation
- First derivative approximation:
  ```
  f'(x) ≈ [f(x+h) - f(x-h)]/(2h) + O(h²)
  ```
- Error is proportional to h²
- For grid array F, derivative at point i:
  ```
  FD[i] = [F[i+1] - F[i-1]]/(2h)
  ```
- Can be rewritten as:
  ```
  FD[i] = (-1/(2h))F[i-1] + (1/(2h))F[i+1]
  ```

### Stencil Patterns
1. 1D Stencils:
   - 3-point stencil (order 1): Uses points [i-1, i, i+1]
   - 5-point stencil (order 2): Uses points [i-2, i-1, i, i+1, i+2]
   - 7-point stencil (order 3): Uses points [i-3, i-2, i-1, i, i+1, i+2, i+3]

2. 2D Stencils:
   - 5-point stencil (order 1): Center point plus 4 neighbors on axes
   - 9-point stencil (order 2): Center point plus 8 surrounding points

3. 3D Stencils:
   - 7-point stencil (order 1): Center point plus 6 neighbors on axes
   - 13-point stencil (order 2): More complex pattern

### Stencil Order
- The number of grid points on each side of the center point
- Reflects the order of the derivative being approximated
- Higher order stencils are used for higher derivatives

### Stencil Sweep
- The computation pattern where a stencil is applied to all relevant input grid points
- Generates output values at all grid points


## 8.2 Parallel stencil: a basic algorithm

We will first present a basic kernel for stencil sweep. For simplicity we assume that there is no dependence between output grid points when generating the output grid point values within a stencil sweep. We further assume that the grid point values on the boundaries store boundary conditions and will not change from input to output. That is, the shaded inside area in the output grid will be calculated, whereas the unshaded boundary cells will remain the same as the input values. This is a reasonable assumption, since stencils are used mainly to solve differential equations with boundary conditions.

This basic kernel performs a stencil sweep. It assumes that each thread block is responsible for calculating a tile of output grid values and that each thread is assigned to one of the output grid points. Since most real-world applications solve three-dimensional (3D) differential equations, the kernel assumes a 3D grid and a 3D seven-point stencil.

The assignment of threads to grid points is done with linear expressions involving the x, y, and z fields of blockIdx, blockDim, and threadIdx. Once each thread has been assigned to a 3D grid point, the input values at that grid point and all neighboring grid points are multiplied by different coefficients (c0 to c6) and added. The values of these coefficients depend on the differential equation that is being solved.

For floating-point to global memory access ratio calculations:
- Each thread performs 13 floating-point operations (seven multiplications and six additions)
- Each thread loads seven input values that are 4 bytes each
- Therefore the ratio is 13/(7×4) = 0.46 OP/B (operations per byte)

This ratio needs to be much larger for the performance of the kernel to be reasonably close to the level supported by the arithmetic compute resources. We will need to use tiling techniques to elevate this ratio.

```cuda
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k]
                                + c1 * in[i * N * N + j * N + (k - 1)]
                                + c2 * in[i * N * N + j * N + (k + 1)]
                                + c3 * in[i * N * N + (j - 1) * N + k]
                                + c4 * in[i * N * N + (j + 1) * N + k]
                                + c5 * in[(i - 1) * N * N + j * N + k]
                                + c6 * in[(i + 1) * N * N + j * N + k];
    }
}
```

## 8.3 Shared memory tiling for stencil sweep

The ratio of floating-point operations to global memory accessing operations can be significantly elevated with shared memory tiling. The design of shared memory tiling for stencils is almost identical to that of convolution, with a few subtle but important differences.

For stencils versus convolution:
- The input tiles of stencils do not include corner grid points (for point stencils)
- Input data reuse in a 2D five-point stencil is significantly lower than in a 3×3 convolution
- For a 2D five-point stencil, the upper bound on the arithmetic to global memory access ratio is only 2.5 OP/B (compared to 4.5 OP/B for 3×3 convolution)

The difference becomes more pronounced with higher dimensions and higher order stencils:
- 2D order-2 stencil (nine-point): 4.5 OP/B vs. 12.5 OP/B for 5×5 convolution
- 2D order-3 stencil (13-point): 6.5 OP/B vs. 24.5 OP/B for 7×7 convolution
- 3D order-3 stencil (19-point): 9.5 OP/B vs. 171.5 OP/B for 7×7×7 convolution

This means the benefit of loading input grid point values into shared memory for stencil sweep can be significantly lower than for convolution, especially for 3D cases, which motivates the use of thread coarsening and register tiling.

In the tiled stencil sweep kernel:
- Each block is of the same size as input tiles
- Some threads are turned off when calculating output grid point values
- The kernel first calculates the beginning x, y, and z coordinates of the input patch for each thread
- An array in shared memory holds the input tile for each block
- Every thread loads one input element
- Guard conditions prevent out-of-bound accesses
- Threads collaborate to load the input tile, then synchronize with a barrier
- Only threads whose output points fall within the output tile calculate results

For evaluating shared memory tiling effectiveness:
- If each input tile is a cube with T grid points in each dimension
- Each output tile has (T-2) grid points in each dimension
- Each block has (T-2)³ active threads
- Each active thread performs 13 floating-point operations
- The floating-point to global memory access ratio = (13(T-2)³)/(4T³) = (13/4)((1-2/T)³)

As T increases, the ratio approaches 13/4 = 3.25 OP/B, but hardware constraints limit T:
- The 1024 thread block size limit makes the practical limit on T around 8 (512 threads)
- Shared memory usage is proportional to T³, limiting tile size
- For T=8, the ratio is only 1.37 OP/B, far from the 3.25 OP/B upper bound

Two major disadvantages of small tile sizes:
1. Limited reuse ratio due to high proportion of halo elements (58% for 8×8×8 3D tile)
2. Adverse impact on memory coalescing (warps accessing distant memory locations)

These limitations motivate the thread coarsening approach covered in the next section.

```cuda
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z >= 1 && threadIdx.y >= 1 && threadIdx.x >= 1
            && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                    + c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                    + c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                                    + c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                    + c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                    + c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                    + c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}
```

## 8.4 Thread coarsening

```cuda
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = c0 * inCurr_s[threadIdx.y][threadIdx.x]
                                        + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
                                        + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                                        + c3 * inCurr_s[threadIdx.y - 1][threadIdx.x]
                                        + c4 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                                        + c5 * inPrev_s[threadIdx.y][threadIdx.x]
                                        + c6 * inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}
```

## 8.5 Register tiling

```cuda
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = c0 * inCurr
                                        + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
                                        + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                                        + c3 * inCurr_s[threadIdx.y - 1][threadIdx.x]
                                        + c4 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                                        + c5 * inPrev
                                        + c6 * inNext;
            }
        }
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}
```

## Exercises

1. Consider a 3D stencil computation on a grid of size 120x120x120, including boundary cells.
    a. What is the number of output grid points that is computed during each stencil sweep?
        The number of output grid points will be the input grid with boundary cells - total boundary cells. 
        We need to generalize this so it carries for arbitrary stencil orders, i.e., n-derivative ops. 
        I'm a fucking idiot gah damn.
        just subtract the fucking stencil order from each dim holy shit dude. 
        total output points = N^3 - (N-order)^3

    b. For the basic kernel in Fig. 8.6, what is the number of thread blocks that are needed, assuming a block size of 8x8x8?
    120/8 = 15 and then cube it for each dim so 3,375 total thread blocks. an 8x8x8 block will consume 512 threads + another 512 threads for the input/stencil which maxes out an entire block.
    meanining we need an entire block for each section of the input grid.

    c. For the kernel with shared memory tiling in Fig. 8.8, what is the number of thread blocks that are needed, assuming a block size of 8^3?
    here we'll have output tile dim with length 6 so then we need to take the output grid size / output tile size = 118 / 6 = 20 if we take ceil and then we need that many blocks in each dim so 20^3 blocks

    d. For the kernel with shared memory tiling and thread coarsening in Fig. 8.10, what is the number of thread blocks that are needed, assuming a block size of 32^2?
    Here we have a block size of 32^2 and we're iterating through z dim with x-y planes and this iterating through the z dim has an output tile size of 6
    so we'll have (118/30)^2 for the x-y plane and then we'll have 118/6 for the z dim giving us (118/30)^2 * (118/6) = 320 blocks

    Notes:
    Now, its really interesting to see that the stencil op is memory bound, not computation bound. Here the naive stencil op only uses 3k blocks while the shared memory tiling uses 8k blocks, but
    the shared memory is far faster.
    The naive implementation has each thread load all 7 points of the stencil from global for each element (7 * 118^3) loads
    While the shared mem tiling kernel makes 1 load per thread totaling to 8,000 * 512 = 4M loads. 

2. Consider an implementation of a seven-point (3D) stencil with shared memory tiling and thread coarsening applied. The implementation is similar to those in Figs 8.10 and 8.12, except that the tiles are not perfect cubes. Instead, a thread block size of 32x32 is used as well as a coarsening factor of 16 (i.e., each thread block processes 16 consecutive output planes in the z dim). 
    a. What is the size of the input tile (in number of elements) that the thread block loads throughout its lifetime?
    32*32*18 = 18k

    b. What is the size of the output tile (in number of elements) that the thread block processes throughout its lifetime?
    30*30*14 = 12k

    c. What is the floating point to global memory access ratio (in OP/B) of the kernel?
    so a 7 order stencil has 7 mulitplications and 6 additions so 13 flops * 12k = 163k
    each thread loads 32*32*18 = 18k elements * 4bytes per element = 70k bytes
    OP/B = 163k / 70k = 2.2 OP/B

    d. How much shared memory (in bytes) is needed by each thread block if register tiling is not used, as in Fig 8.10?
    you need 3 xy planes of 32^2 to compute each output element.
    so 3 * 32^2 * 4bytes = 12,288 bytes = 12KB

    e. How much shared memory (in bytes) is needed by each thread block if register tiling is used, as in Fig 8.12?
    this will just be 1 plane of 32^2 since the current plane is the only one in shared memory
    32^2 * 4bytes = 4,096 bytes = 4KB