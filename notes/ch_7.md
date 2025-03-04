# 7 Convolution

> 💡 **Core Concept**: Convolution is a fundamental array operation where each output element is calculated as a weighted sum of corresponding input elements and their neighbors, widely used in signal processing, image processing, and deep learning.

## 7.1 Background

Convolution is an array operation where each output element is a weighted sum of the corresponding input element and surrounding elements. The weights used in this calculation are defined by a **filter array**, also known as a convolution kernel. To avoid confusion with CUDA kernel functions, we'll refer to these as **convolution filters**.

### Types of Convolution

Convolution can be performed on data of different dimensionality:
- **1D convolution**: For audio signals (samples over time)
- **2D convolution**: For images (pixels in x-y space)
- **3D convolution**: For video or volumetric data
- And higher dimensions

### Mathematical Definition

For a **1D convolution** with:
- Input array: [x₀, x₁, ..., xₙ₋₁]
- Filter array of size (2r+1): [f₀, f₁, ..., f₂ᵣ] where r is the filter radius
- Output array: y

The convolution is defined as:
```
yᵢ = ∑ⱼ₌₋ᵣʳ fᵢ₊ⱼ × xᵢ
```

The filter is typically symmetric around the center element, with r elements on each side.

### Example: 1D Convolution

For a filter with radius r=2 (5 elements total) applied to a 7-element array:

- Input array x = [8, 2, 5, 4, 1, 7, 3]
- Filter f = [1, 3, 5, 3, 1]

To calculate y[2]:
```
y[2] = f[0]×x[0] + f[1]×x[1] + f[2]×x[2] + f[3]×x[3] + f[4]×x[4]
     = 1×8 + 3×2 + 5×5 + 3×4 + 1×1
     = 52
```

To calculate y[3]:
```
y[3] = f[0]×x[1] + f[1]×x[2] + f[2]×x[3] + f[3]×x[4] + f[4]×x[5]
     = 1×2 + 3×5 + 5×4 + 3×1 + 1×7
     = 47
```

> 🔍 **Insight**: Each output element calculation can be viewed as an inner product between the filter array and a window of the input array centered at the corresponding position.

### Handling Boundary Conditions

When calculating output elements near array boundaries, we need to handle **ghost cells** - elements that would fall outside the input array:

- Common approach: Assign default value (typically 0) to missing elements
- For audio: Assume signal volume is 0 before recording starts
- For images: Various strategies (zeros, edge replication, etc.)

Example calculation at boundary (y[1]):
```
y[1] = f[0]×0 + f[1]×x[0] + f[2]×x[1] + f[3]×x[2] + f[4]×x[3]
     = 1×0 + 3×8 + 5×2 + 3×5 + 1×4
     = 53
```

### 2D Convolution

For image processing and computer vision, we use 2D convolution:

- The filter becomes a 2D array with dimensions (2rₓ+1) × (2rᵧ+1)
- Each output element is calculated by:
  ```
  P[y,x] = ∑ⱼ₌₋ᵣᵧʳʸ ∑ₖ₌₋ᵣₓʳˣ f[y+j,x+k] × N[y,x]
  ```

**Example**: For a 5×5 filter (rₓ = rᵧ = 2):
1. Take a 5×5 subarray from input centered at the position being calculated
2. Perform element-wise multiplication with the filter
3. Sum all resulting products to get the output element

> ⚠️ **Important**: 2D convolution has more complex boundary conditions (horizontal, vertical, or both). Different applications handle these boundaries differently.

## 7.2 Parallel convolution: a basic algorithm

> 💡 **Core Concept**: Convolution is ideally suited for parallel computing since each output element can be calculated independently, allowing efficient mapping to CUDA threads.

### Parallelization Approach

The independence of output element calculations makes convolution a perfect fit for GPU parallelization:

- Each thread calculates one output element
- Threads are organized in a 2D grid to match the 2D output structure
- For larger images, we divide the calculation into blocks of threads

### Kernel Implementation

The basic 2D convolution kernel takes the following parameters:
- Input array `N`
- Filter array `F`
- Output array `P`
- Filter radius `r`
- Image dimensions (`width` and `height`)

For a 2D convolution with a square filter:

```c
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1, fRow++) {
        for (int fCol = 0, fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow][fCol] * N[inRow*width + inCol];
            }
        }
    }
    P[outRow][outCol] = Pvalue;
}
```

### Thread Mapping and Execution

The mapping from threads to output elements is straightforward:
- Each thread calculates one output element at position `(outRow, outCol)`
- Each thread block processes a tile of the output
- For each output element, we need a window of input elements centered at the corresponding position

> 📝 **Example**: If using 4×4 thread blocks for a 16×16 image, we would have a 4×4 grid of blocks. Thread (1,1) in block (1,1) would compute output element P[5][5].

### Handling Boundary Conditions

The kernel handles boundary conditions with an if-statement:
- Checks if the required input element is within bounds
- Skips multiplication for out-of-bounds elements (ghost cells)
- This approach assumes ghost cells have value 0

> 🔍 **Insight**: The if-statement causes control flow divergence, especially for threads computing output elements near image edges. For large images with small filters, this divergence has minimal impact on performance.

### Performance Considerations

This basic implementation faces two main challenges:

1. **Control Flow Divergence**:
   - Threads computing boundary pixels take different paths through the if-statement
   - Impact is minimal for large images with small filters

2. **Memory Bandwidth Limitations**:
   - Low arithmetic intensity: Only ~0.25 operations per byte (2 operations for every 8 bytes loaded)
   - Global memory access is a major bottleneck
   - Performance is far below peak capability

> ⚠️ **Optimization Needed**: This naive implementation is memory-bound. Advanced techniques like constant memory and tiling can significantly improve performance by reducing global memory accesses.

## 7.3 Constant memory and caching

> 💡 **Core Concept**: Filter arrays in convolution have properties that make them ideal candidates for CUDA's constant memory, which provides high-bandwidth access through specialized caching, significantly improving performance.

### Filter Properties for Constant Memory

The filter array `F` used in convolution has three key properties that make it well-suited for constant memory:

1. **Small Size**: 
   - Most convolution filters have a radius ≤ 7
   - Even 3D filters typically contain ≤ 343 elements (7³)

2. **Constant Contents**:
   - The filter values remain unchanged throughout kernel execution

3. **Consistent Access Pattern**:
   - All threads access the filter elements
   - They access elements in the same order (starting from F[0][0])
   - Access pattern is independent of thread indices

### CUDA Constant Memory Overview

Constant memory in CUDA:
- Located in device DRAM (like global memory)
- Limited to 64KB total size
- Read-only during kernel execution
- Visible to all thread blocks
- Aggressively cached in specialized hardware

#### Modified Kernel Using Constant Memory:

```c
// Define filter size at compile time
#define FILTER_RADIUS 3

// Declare filter in constant memory
__constant__ float F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

// Host code with filter already initialized in F_h
cudaMemcpyToSymbol(F, F_h, sizeof(float)*(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1));

__global__ void convolution_2D_constant_kernel(float *N, float *P, int width, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    
    for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
        for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                // F is now accessed as a global variable
                Pvalue += F[fRow*(2*FILTER_RADIUS+1)+fCol] * N[inRow*width + inCol];
            }
        }
    }
    
    P[outRow*width + outCol] = Pvalue;
}
```

> 📝 **Note**: The filter is now accessed as a global variable rather than through a function parameter.

### Cache Hierarchy in Modern Processors

Modern processors use a hierarchy of cache memories to mitigate DRAM bottlenecks:

- **L1 Cache**: 
  - Closest to processor cores
  - Small (16-64KB) but very fast
  - Typically per-core or per-SM

- **L2 Cache**:
  - Larger (hundreds of KB to few MB)
  - Slower than L1 (tens of cycles latency)
  - Often shared among multiple cores/SMs

- **Cache vs. Shared Memory**:
  - Caches are **transparent** (automatic) to programs
  - Shared memory requires explicit declaration and management

### Constant Caches

GPUs implement specialized **constant caches** for constant memory:

- Optimized for read-only access patterns
- Designed for efficient area and power usage
- Extremely effective when all threads in a warp access the same memory location
- Well-suited for convolution filters where access patterns are uniform across threads

> 🔍 **Insight**: When all threads access F elements in the same pattern, the constant cache provides tremendous bandwidth without consuming DRAM resources.

### Performance Impact

Using constant memory for the filter array:
- Effectively eliminates DRAM bandwidth consumption for filter elements
- Doubles the arithmetic intensity from ~0.25 to ~0.5 OP/B
- Each thread now only needs to access input array elements from global memory

> ⚠️ **Remember**: While constant memory optimizes filter access, input array accesses still consume significant memory bandwidth. Further optimizations for input array access are needed.


## 7.4 Tiled convolution with halo cells

> 💡 **Core Concept**: Tiled convolution addresses memory bandwidth bottlenecks by having threads collaborate to load input elements into shared memory, significantly reducing global memory accesses and improving arithmetic intensity.

### Input and Output Tiles

Understanding the relationship between input and output tiles is crucial for tiled convolution:

- **Output Tile**: Collection of output elements processed by each thread block
- **Input Tile**: Collection of input elements needed to calculate an output tile
  - Must be extended by the filter radius in each direction
  - Includes "halo cells" - elements outside the output tile but needed for calculations

> 📝 **Example**: For a 5×5 filter (radius=2) and a 4×4 output tile, the input tile would be 8×8 (extended by 2 in each direction), making it 4× larger than the output tile.

### Input vs. Output Tile Size Relationship

- Input tiles are always larger than output tiles due to halo cells
- The size difference depends on filter radius and output tile dimensions
- For practical output tile sizes, the ratio is typically closer to 1.0
  - Example: 16×16 output tile with 5×5 filter requires 20×20 input tile (ratio of 1.6)

> 🔍 **Insight**: Even with practical tile sizes, input tiles can be significantly larger than output tiles, complicating kernel design.

### Thread Organization Strategies

Two approaches to handle the discrepancy between input and output tile sizes:

1. **Match Thread Blocks to Input Tiles**:
   - Simplifies input loading (one thread per input element)
   - Requires disabling threads during output calculation
   - Reduces execution resource utilization

2. **Match Thread Blocks to Output Tiles**:
   - Complicates input loading (threads must iterate to load all elements)
   - Simplifies output calculation (all threads participate)
   - Better utilizes execution resources

### Tiled Convolution Kernel Implementation

```c
__global__ void convolution_2D_tiled_kernel(float *N, float *P, int width, int height) {
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    
    // Load input tile into shared memory
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Calculate output elements
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;
    
    if (tileRow >= 0 && tileRow < OUT_TILE_DIM && 
        tileCol >= 0 && tileCol < OUT_TILE_DIM) {
        
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                Pvalue += F[fRow*(2*FILTER_RADIUS+1)+fCol] * 
                          N_s[tileRow+fRow][tileCol+fCol];
            }
        }
        
        if (row < height && col < width) {
            P[row*width + col] = Pvalue;
        }
    }
}
```

### Thread Mapping for Output Calculation

- Only a subset of threads calculate output elements
- Typically deactivate `FILTER_RADIUS` exterior layers of threads
- Active thread (tx, ty) calculates output element (tx-FILTER_RADIUS, ty-FILTER_RADIUS)
- Each thread uses a patch of input elements centered at its position

> 📊 **Visualization**: For a 3×3 filter (radius=1) with 8×8 input tiles and thread blocks, only the inner 6×6 threads calculate output elements.

### Performance Analysis: Arithmetic Intensity

The tiled algorithm significantly improves arithmetic intensity:

- **Arithmetic Operations**: `OUT_TILE_DIM²` × `(2×FILTER_RADIUS+1)²` × 2
- **Global Memory Accesses**: `(OUT_TILE_DIM+2×FILTER_RADIUS)²` × 4 bytes
- **Arithmetic Intensity Ratio**: 
  ```
  OUT_TILE_DIM² × (2×FILTER_RADIUS+1)² × 2
  ----------------------------------------
  (OUT_TILE_DIM+2×FILTER_RADIUS)² × 4
  ```

> 📈 **Example**: For a 5×5 filter and 32×32 input tiles (28×28 output tiles), the ratio is 9.57 OP/B, compared to 0.5 OP/B in the basic algorithm.

### Asymptotic Analysis

- For very large tiles, the ratio approaches `(2×FILTER_RADIUS+1)² × 2/4`
- Larger filters yield higher arithmetic intensity
- Practical limitations (32×32 thread blocks) prevent reaching theoretical bounds:
  - 5×5 filter: 9.57 OP/B achieved vs. 12.5 OP/B theoretical
  - 9×9 filter: 22.78 OP/B achieved vs. 40.5 OP/B theoretical

> ⚠️ **Warning**: Small tile sizes may result in significantly less reduction in memory accesses than expected. For example, 8×8 tiles with 5×5 filters achieve only 3.13 OP/B.

## 7.5 Tiled convolution using caches for halo cells

> 💡 **Core Concept**: Modern GPUs can leverage L2 caches to efficiently handle halo cells, allowing for simplified tiled convolution implementations that load only internal tile elements into shared memory.

### Leveraging Hardware Caches

- Halo cells of one tile are internal elements of neighboring tiles
- By the time a block needs its halo cells, they may already be in L2 cache
- This allows a simplified approach where:
  - Input and output tiles have the same dimensions
  - Only internal elements are loaded into shared memory
  - Halo cells are accessed directly from global memory (likely served from cache)

### Simplified Kernel Implementation

```c
__global__ void convolution_2D_tiled_caching_kernel(float *N, float *P, int width, int height) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    
    // Load only internal elements to shared memory
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    }
    __syncthreads();
    
    float Pvalue = 0.0f;
    
    if (row < height && col < width) {
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                int inRow = row - FILTER_RADIUS + fRow;
                int inCol = col - FILTER_RADIUS + fCol;
                
                // Check if element is within the tile (use shared memory)
                if (inRow >= blockIdx.y*TILE_DIM && inRow < (blockIdx.y+1)*TILE_DIM &&
                    inCol >= blockIdx.x*TILE_DIM && inCol < (blockIdx.x+1)*TILE_DIM) {
                    
                    Pvalue += F[fRow*(2*FILTER_RADIUS+1)+fCol] * 
                              N_s[inRow-blockIdx.y*TILE_DIM][inCol-blockIdx.x*TILE_DIM];
                }
                // Check if element is a valid halo cell (use global memory)
                else if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    Pvalue += F[fRow*(2*FILTER_RADIUS+1)+fCol] * N[inRow*width + inCol];
                }
                // Ghost cells are assumed to be zero (no action needed)
            }
        }
        P[row*width + col] = Pvalue;
    }
}
```

### Advantages of Cache-Based Approach

1. **Simplified Tile Management**:
   - Input tiles, output tiles, and thread blocks can all have the same dimensions
   - Tile dimensions can be powers of 2, which is more efficient for memory access

2. **Reduced Memory Divergence**:
   - More uniform memory access patterns
   - Less control divergence during execution

3. **Efficient Use of Hardware Resources**:
   - Leverages hardware caches for data reuse
   - Reduces shared memory requirements

> 🔍 **Insight**: This approach represents a hybrid strategy that combines explicit data management (shared memory for internal elements) with implicit caching (L2 cache for halo cells).

## 7.6 Summary

> 💡 **Core Concept**: Convolution represents a fundamental parallel computation pattern that appears in many applications and can be optimized through various techniques to overcome memory bandwidth limitations.

### Key Concepts Covered

1. **Convolution as a Parallel Pattern**:
   - Fundamental operation in computer vision, signal processing, and deep learning
   - Represents a general pattern found in many parallel algorithms
   - Related to stencil algorithms and grid-based computations

2. **Optimization Progression**:
   - Basic parallel algorithm (memory-bound)
   - Constant memory optimization for filter elements
   - Tiled algorithm with shared memory for input elements
   - Cache-based tiled algorithm for simplified implementation

3. **Performance Analysis**:
   - Arithmetic intensity as a key metric
   - Impact of tile size and filter size on performance
   - Theoretical bounds vs. practical limitations

### Applications Beyond This Chapter

Convolution techniques apply to many other computational patterns:
- Stencil algorithms in partial differential equation solvers (Chapter 8)
- Grid point force/potential calculations
- Convolutional neural networks (Chapter 16)
- Iterative MRI reconstruction (Chapter 17)

### Extending to Higher Dimensions

The techniques presented for 1D and 2D convolutions extend to 3D and higher:
- More complex index calculations
- Additional loop nesting for higher dimensions
- Same fundamental principles of memory optimization

> 📚 **Further Learning**: The reader is encouraged to implement higher-dimension kernels as exercises to reinforce understanding of these concepts.

## Exercises

1. Calculate the P[0] value in Fig. 7.3
x = [8, 2, 5, 4, 1, 7, 3]
f = [1, 3, 5, 3, 1]
P[0] = (1*0) + (3*0) + (5*8) + (2*3) + (5*1) = 51

2. Consider performing a 1D convolution on an array N = [4, 1, 3, 2, 3] with filter F = [2, 1, 4]
x = [8, 21, 13, 20, 7]

3. What do you think the following 1D convolution filters are doing?
a. [0, 1, 0] - identity
b. [0, 0, 1] - shift values to the right
c. [1, 0, 0] - shift values to the left
d. [-1/2, 0, 1/2] - smooth and translate down
e. [1/3, 1/3, 1/3] - smooth and compress about mean value

4. Consider performing a 1D convolution on an array of size N with a filter of size M:
a. How many ghost cells are there in total?
M = 2r+1
the amount of ghost cells will be 2r -> 1r on each side of array N
b. How many multiplications are performed if ghost cells are treated as multiplications?
N*M
c. How many multiplications are performed if ghost cells are not treated as multiplications
(N*M) - r*(r+1)

5. Consider performing a 2D convolution on a square matrix of size NxN with a square filter of size MxM:
a. How many ghost cells are there in total?
(N+2r)*(N+2r) - N*N
b. How many multiplications are performed if ghost cells are treated as multiplications?
M^2 * N^2
c. How many mulitplications are performed if ghost cells are not treated as multiplications?
N^2 * M^2 - (N+r)^2 * M^2 + (N+r-M)^2 * M^2

6. Consider performing a 2D convolution on a rectangular matrix of size N1xN2 with a retangular mask of size M1xM2:
a. How many ghost cells are there in total?

b. How many multiplications are performed if ghost cells are treated as multiplications?

c. How many mulitiplications are performed if ghost cells are not treated as multiplications?

7. Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size NxN with a filter of size MxM using an output tile of size TxT.

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                      int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}
```

a. How many thread blocks are needed?
ceil(N / OUT_TILE_DIM)^2

b. How many threads are needed per block?
IN_TILE_DIM^2 = 1024

c. How much shared memory is needed per block?
4,096bytes -> threads * 4bytes

d. Repeat the same questions if you were using the kernel in Fig. 7. 15

```c
#define TILE_DIM 32

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P,
                                                             int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Loading input tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];

    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Calculating output elements
    if (col < width && row < height) {
        float Pvalue = 0.0f;

        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                if (threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
                    threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
                    threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM) {
                    Pvalue += F_c[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
                } else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {
                        Pvalue += F_c[fRow][fCol] * N[(row - FILTER_RADIUS + fRow) * width + (col - FILTER_RADIUS + fCol)];
                    }
                }
            }
        }

        P[row * width + col] = Pvalue;
    }
}
```
  a. How many thread blocks are needed?
  ceil(N/TILE_DIM)^2

  b. How many threads are needed per block?
  32*32 threads

  c. How much shared memory is needed per block?
  32*32*4 = 4,096 bytes