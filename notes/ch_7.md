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

