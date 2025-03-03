# Programming Massively Parallel Processors: A Hands-on Approach

## 3. Multidimensional Grids and Data

> 💡 **Core Concept**: This chapter explains how to organize CUDA threads into 2D/3D structures to efficiently process multidimensional data like matrices and images.

### 3.1 Multidimensional Thread Organization

#### Block and Grid Dimensionality

CUDA supports organizing threads into one, two, or three dimensions using the `dim3` type, which allows for:
- More intuitive mapping to multidimensional data
- Better spatial locality for certain algorithms
- More natural expression of problems with 2D/3D structure

#### Thread Block Limitations

```c
// Thread block with different configurations
dim3 block1D(256, 1, 1);        // 256 threads in x-dimension
dim3 block2D(16, 16, 1);        // 256 threads in x,y-dimensions
dim3 block3D(8, 8, 4);          // 256 threads in x,y,z-dimensions
```

> ⚠️ **Important**: The total number of threads in a block must not exceed 1024 in current CUDA systems.

Valid configurations include:
- `(512, 1, 1)` = 512 threads
- `(8, 16, 4)` = 512 threads
- `(32, 16, 2)` = 1024 threads

Invalid configuration:
- `(32, 32, 2)` = 2048 threads (exceeds 1024 limit)

#### Creating Grids with Different Dimensionality

Grid and block dimensions don't need to match. You can have:
- 2D grid of 1D blocks
- 1D grid of 3D blocks
- Any other combination

```c
// Creating a 2D grid of 3D blocks
dim3 gridDim(2, 2, 1);      // 2×2×1 = 4 blocks in grid
dim3 blockDim(4, 2, 2);     // 4×2×2 = 16 threads per block

// Launch kernel with these dimensions
KernelFunction<<<gridDim, blockDim>>>(...);
```

Visualization of this grid:
```
                 Block (0,0)         Block (1,0)
                ┌───────────┐       ┌───────────┐
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                └───────────┘       └───────────┘

                 Block (0,1)         Block (1,1)
                ┌───────────┐       ┌───────────┐
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                └───────────┘       └───────────┘
```

#### Thread Indexing in Multidimensions

With multidimensional organization, we access thread coordinates using:

```c
// Thread coordinates within a block
int threadX = threadIdx.x;
int threadY = threadIdx.y; 
int threadZ = threadIdx.z;

// Block coordinates within the grid
int blockX = blockIdx.x;
int blockY = blockIdx.y;
int blockZ = blockIdx.z;
```

For global 2D coordinates:
```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 3.2 Mapping Threads to Multidimensional Data

#### Linearizing Multidimensional Arrays

Modern computers use "flat" memory, so multidimensional arrays must be linearized.

##### Row-Major vs. Column-Major Order

**Row-Major Order** (C/C++ default):
- Elements in the same row are stored consecutively
- Moving horizontally in the array accesses adjacent memory locations
- Used in C, C++, Python, and most other languages

**Column-Major Order** (Fortran, MATLAB):
- Elements in the same column are stored consecutively
- Moving vertically in the array accesses adjacent memory locations
- Used in Fortran, R, MATLAB

Example 2×3 matrix:
```
[ 1  2  3 ]
[ 4  5  6 ]
```

Row-major representation: `[1, 2, 3, 4, 5, 6]`
Column-major representation: `[1, 4, 2, 5, 3, 6]`

#### Row-Major Linear Indexing Formulas

For a 2D array with dimensions `width × height`:
```c
// Access element at (row, col)
int index = row * width + col;
```

For a 3D array with dimensions `width × height × depth`:
```c
// Access element at (x, y, z)
int index = z * (height * width) + y * width + x;
```

> 🔍 **Tip**: Always use variables for the dimensions rather than hardcoding them, making your code easier to maintain and adapt.

#### Image Processing Example: RGB to Grayscale

The following kernel converts an RGB image to grayscale:

```c
#define CHANNELS 3  // RGB channels

__global__ void colorToGrayscaleConversion(
                unsigned char* Pout,  // Output grayscale image
                unsigned char* Pin,   // Input RGB image
                int width,            // Image width
                int height) {         // Image height
    
    // Calculate 2D position in the image
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image boundaries
    if (col < width && row < height) {
        // Calculate 1D offset for grayscale image
        int grayOffset = row * width + col;
        
        // Calculate 1D offset for RGB image (3 channels per pixel)
        int rgbOffset = grayOffset * CHANNELS;
        
        // Extract RGB components
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        
        // Standard luminance conversion formula
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}
```

To launch this kernel for a 1024×768 image:

```c
// Choose a 16×16 block size (common for 2D processing)
dim3 blockSize(16, 16);

// Calculate grid dimensions to cover the entire image
dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
              (height + blockSize.y - 1) / blockSize.y);

// Launch kernel
colorToGrayscaleConversion<<<gridSize, blockSize>>>(d_grayImage, d_rgbImage, width, height);
```

> 🔑 **Key insight**: The boundary check `if (col < width && row < height)` is crucial since we're launching more threads than pixels to ensure coverage of the entire image.

### 3.3 Image Blur: A More Complex Kernel

Image blur is a common operation that requires each thread to access surrounding pixels, demonstrating a more complex access pattern.

#### Blur Kernel Implementation

```c
#define BLUR_SIZE 1  // Radius of blur kernel (1=3×3 filter)

__global__ void blurKernel(unsigned char* in,    // Input image
                         unsigned char* out,    // Output image
                         int width, 
                         int height) {
    
    // Calculate pixel position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int pixVal = 0;     // Sum of pixel values
        int pixels = 0;     // Count of pixels in average
        
        // Blur kernel is (2*BLUR_SIZE+1) × (2*BLUR_SIZE+1)
        // For BLUR_SIZE=1, we have a 3×3 kernel
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                
                // Calculate position of neighboring pixel
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                // Check if neighbor is within image boundaries
                if (curRow >= 0 && curRow < height && 
                    curCol >= 0 && curCol < width) {
                    // Add pixel value to sum
                    pixVal += in[curRow * width + curCol];
                    pixels++; // Increment count
                }
            }
        }
        
        // Write average to output image
        out[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```

> 💡 **Understanding**: This implements a box blur by averaging all pixels in a 3×3 neighborhood around each pixel. The boundary check ensures proper handling of image edges.

#### Thread Access Pattern Visualization

For a 3×3 blur filter, each thread's access pattern looks like:

```
┌───┬───┬───┐
│ X │ X │ X │ 
├───┼───┼───┤
│ X │ C │ X │
├───┼───┼───┤
│ X │ X │ X │
└───┴───┴───┘
```

Where `C` is the center pixel and `X` marks neighboring pixels that are accessed to compute the average.

> ⚠️ **Performance note**: This implementation has suboptimal memory access patterns. In Chapter 5, we'll learn how to use shared memory to optimize operations like this where multiple threads need access to the same data.

### 3.4 Matrix Multiplication

Matrix multiplication is a fundamental operation in scientific computing and demonstrates the power of 2D thread organization.

#### Matrix Multiplication Algorithm

For matrices:
- `M` (dimensions `m × n`)
- `N` (dimensions `n × k`)
- `P` (dimensions `m × k`) as the result of `M × N`

The formula for each element of P is:
```
P[i,j] = ∑(k=0 to n-1) M[i,k] × N[k,j]
```

#### Basic Matrix Multiplication Kernel

```c
__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
    // Calculate row and column indices of P
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix dimensions
    if (row < width && col < width) {
        float Pvalue = 0;
        
        // Multiply row of M by column of N
        for (int k = 0; k < width; ++k) {
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        
        // Store result in P
        P[row * width + col] = Pvalue;
    }
}
```

> 📝 **Note**: This simplified example assumes square matrices of the same width for clarity. A more general implementation would handle matrices of different dimensions.

#### Launch Configuration for Matrix Multiplication

```c
// For a 1024×1024 matrix
int width = 1024;
int blockSize = 16; // 16×16 = 256 threads per block

// Calculate grid dimensions
dim3 dimBlock(blockSize, blockSize);
dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
             (width + dimBlock.y - 1) / dimBlock.y);

// Launch kernel
matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
```

#### Memory Access Pattern Analysis

This kernel has different memory access patterns for matrices M and N:

- For matrix M: Row-wise access (coalesced memory access)
- For matrix N: Column-wise access (non-coalesced, can be inefficient)

Visualization for one thread computing P[i,j]:

```
Matrix M:           Matrix N:           Matrix P:
┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐
│   │   │   │   │   │   │ ↓ │   │   │   │   │   │   │   │
├───┼───┼───┼───┤   ├───┼───┼───┼───┤   ├───┼───┼───┼───┤
│ → │ → │ → │ → │   │   │ ↓ │   │   │   │   │   │ X │   │
├───┼───┼───┼───┤   ├───┼───┼───┼───┤   ├───┼───┼───┼───┤
│   │   │   │   │   │   │ ↓ │   │   │   │   │   │   │   │
├───┼───┼───┼───┤   ├───┼───┼───┼───┤   ├───┼───┼───┼───┤
│   │   │   │   │   │   │ ↓ │   │   │   │   │   │   │   │
└───┴───┴───┴───┘   └───┴───┴───┴───┘   └───┴───┴───┴───┘
```

Where `→` shows row access in M, `↓` shows column access in N, and `X` marks the output element being computed.

> 🔍 **Performance insight**: This implementation is inefficient because:
> 1. It performs redundant global memory accesses
> 2. Column-wise access in matrix N causes non-coalesced memory access
> 
> In Chapter 5, we'll learn to optimize this using shared memory.

### 3.5 Practical Tips for Multidimensional Thread Organization

#### Dimensionality Choice

- Choose dimensionality that naturally matches your data
- For images and matrices, 2D organization is typically best
- For 1D data, stick with 1D organization for simplicity

#### Block Size Selection

For 2D blocks, common choices include:
- 16×16 (256 threads)
- 32×8 (256 threads)
- 8×32 (256 threads)

Factors to consider:
1. Total threads should be a multiple of 32 (warp size)
2. Avoid very small dimensions (e.g., prefer 32×8 over 128×2)
3. Consider memory access patterns (e.g., prefer wider blocks for row-major access)

#### 3D Organization Considerations

- 3D thread organization is useful for volumetric data (e.g., medical imaging)
- Keep the total thread count under 1024
- Common 3D block configurations: 8×8×8, 16×8×4, 32×4×4

### 3.6 Key Takeaways

- CUDA supports 1D, 2D, and 3D thread organization through blockIdx and threadIdx
- Each thread needs to calculate its position in the data using these indices
- Boundary checks are essential when mapping threads to data
- Row-major vs. column-major storage affects memory access efficiency
- Thread organization should match data organization when possible
- Matrix multiplication and image processing benefit significantly from 2D thread organization

***

**Exercise Ideas**:
1. Modify the grayscale conversion to use a different weight formula
2. Implement a Gaussian blur instead of a box blur
3. Implement matrix-vector multiplication using 2D thread organization 