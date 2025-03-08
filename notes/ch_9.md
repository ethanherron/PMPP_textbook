# 9 Parallel histogram

## Chapter Outline
- 9.1 Background
- 9.2 Atomic operations and a basic histogram kernel
- 9.3 Latency and throughput of atomic operations
- 9.4 Privatization
- 9.5 Coarsening
- 9.6 Aggregation
- 9.7 Summary

## Introduction

This chapter introduces the parallel histogram computation pattern, where each output element can potentially be updated by multiple threads, requiring coordination to avoid interference. Unlike previously discussed patterns that follow the "owner-computes" rule, histogram computation involves output interference that must be carefully managed.

Key aspects of parallel histogram:
- Output elements can be updated by any thread
- Requires coordination among threads to avoid interference
- Serves as an example for handling output interference in parallel computation
- Provides optimization techniques applicable to many parallel algorithms

The chapter starts with a baseline approach using atomic operations to serialize updates, then presents optimization techniques like privatization to enhance performance while maintaining correctness.

## 9.1 Background

A histogram displays the frequency or count of data values in a dataset, typically represented as bars where:
- Value intervals are plotted on the horizontal axis
- Count of data values in each interval is represented by bar height

For example, Fig. 9.1 shows a histogram of letters in the phrase "programming massively parallel processors" with value intervals containing four letters each (a-d, e-h, etc.).

Histograms provide useful summaries of datasets that quickly reveal patterns or features:
- Credit card usage patterns can identify potential fraud
- Pixel luminosity histograms help identify objects in computer vision
- Speech recognition systems use histograms for pattern identification
- Website purchase recommendations rely on usage pattern histograms
- Scientific data analysis uses histograms to correlate phenomena

Computing histograms sequentially is straightforward:

```c
void histogram(char* data, int length, int* histo) {
    for (int i = 0; i < length; i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histo[alphabet_position/4]++;
        }
    }
}
```

This sequential algorithm:
- Has O(N) computational complexity
- Accesses data sequentially for good cache utilization
- Uses a small output array that fits in CPU cache
- Is typically memory-bound on modern CPUs

## 9.2 Atomic operations and a basic histogram kernel

The most straightforward approach to parallelizing histogram computation is to launch one thread per input element, with each thread processing its assigned element and incrementing the appropriate counter.

This approach faces a critical challenge: **output interference** - multiple threads may need to update the same counter simultaneously.

### Race Conditions in Read-Modify-Write Operations

Updating a histogram counter involves three steps:
1. Reading the current value (read)
2. Adding one to the value (modify)
3. Writing the new value back (write)

This read-modify-write sequence creates potential race conditions where the outcome depends on the relative timing of operations:

**Example Scenario 1**: Thread 1 completes its update before Thread 2 begins
- histo[x] starts with value 0
- Thread 1: read(0), modify(+1), write(1)
- Thread 2: read(1), modify(+1), write(2)
- Final value is 2 (correct)

**Example Scenario 2**: Thread operations interleave
- histo[x] starts with value 0
- Thread 1: read(0), modify(+1)
- Thread 2: read(0), modify(+1), write(1)
- Thread 1: write(1)
- Final value is 1 (incorrect - Thread 1's update was lost)

These race conditions produce inconsistent and incorrect results.

### Atomic Operations

Atomic operations solve this problem by making the read-modify-write sequence indivisible:
- An atomic operation cannot be interrupted by another atomic operation to the same memory location
- This serializes updates to the same memory location
- Execution order between threads is not enforced, only that operations to the same location do not overlap

CUDA provides atomic operations like `atomicAdd()` that implement this behavior:

```cuda
__global__ void histogram_kernel(char* data, int length, int* histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[alphabet_position/4]), 1);
        }
    }
}
```

The key changes from sequential code:
1. Thread index calculation replaces the loop
2. Regular increments are replaced with `atomicAdd()` calls

## 9.3 Latency and throughput of atomic operations

While atomic operations ensure correctness, they can significantly impact performance by serializing portions of the parallel execution.

### Performance Impact of Atomic Operations

Atomic operations to the same memory location cannot overlap, which means:
- Only one operation can be in progress at a time
- Each operation must complete before the next can begin
- The duration of each operation is approximately the memory load latency plus store latency

This serialization creates a throughput bottleneck:

```
Atomic throughput = 1 / (memory access latency)
```

**Example calculation**:
- Memory system: 128 GB/s peak bandwidth, 200-cycle access latency, 1 GHz clock
- Peak access throughput: 32 billion elements/second
- Atomic throughput to a single location: 2.5 million operations/second (1/(400 cycles) × 1G cycles/second)

The actual throughput depends on contention patterns:
- If atomic operations are evenly distributed across histogram bins, throughput increases
- In practice, data distributions are often biased, creating hotspots
- Example: In "programming massively parallel processors", letters are concentrated in m-p and q-t intervals

### Hardware Improvements

Modern GPUs improve atomic operation performance by:
- Supporting atomic operations in the last-level cache 
- Reducing access latency from hundreds of cycles to tens of cycles
- Keeping frequently accessed variables in cache
- Improving atomic throughput by an order of magnitude compared to early GPU generations

## 9.4 Privatization

Privatization is a technique to alleviate contention by redirecting traffic away from heavily contended locations. The approach involves:
- Creating multiple private copies of the output data structure
- Assigning each subset of threads to update its own private copy
- Merging private copies into the final result at the end

Advantages of privatization:
- Dramatically reduces contention for individual memory locations
- Allows private copies to be accessed at much lower latency
- Significantly increases throughput for data structure updates

The key trade-off is balancing contention reduction against the overhead of merging private copies.

### Implementation Strategies

Common privatization approaches:
1. Create fixed number of private copies (e.g., 2, 4, 8)
2. Assign thread blocks to specific copies based on block index
3. Create a private copy for each thread block (most common approach)

The thread-block approach offers several advantages:
- Threads within a block can synchronize using `__syncthreads()`
- Private copies can be placed in shared memory
- Merging can be handled efficiently

### Global Memory Implementation

```cuda
__global__ void histogram_privatized_kernel(char* data, int length, int* histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[blockIdx.x * NUM_BINS + alphabet_position/4]), 1);
        }
    }
    
    __syncthreads();
    
    for (int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
        int privateVal = histo[blockIdx.x * NUM_BINS + binIdx];
        if (privateVal > 0) {
            atomicAdd(&(histo[binIdx]), privateVal);
        }
    }
}
```

In this implementation:
- Each block updates its private copy in global memory (offset by `blockIdx.x * NUM_BINS`)
- After all threads finish updating, they merge results into the first `NUM_BINS` elements
- Contention is reduced by a factor approximately equal to the number of active blocks

### Shared Memory Implementation

```cuda
__global__ void histogram_privatized_shared_kernel(char* data, int length, int* histo) {
    __shared__ int histo_s[NUM_BINS];
    
    for (int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    
    __syncthreads();
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }
    
    __syncthreads();
    
    for (int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
        int privateVal = histo_s[binIdx];
        if (privateVal > 0) {
            atomicAdd(&(histo[binIdx]), privateVal);
        }
    }
}
```

Key advantages of shared memory implementation:
- Access latency is dramatically reduced (few cycles vs. hundreds of cycles)
- Higher throughput for atomic operations
- Eliminates cache pollution in the L2 cache

## 9.5 Coarsening

Privatization introduces overhead during the merging phase, which increases with the number of thread blocks. Thread coarsening can reduce this overhead by:
- Reducing the number of thread blocks
- Having each thread process multiple input elements
- Maintaining the same total computational workload

Two main strategies for coarsening are available:

### Contiguous Partitioning

```cuda
__global__ void histogram_coarsening_contiguous(char* data, int length, int* histo) {
    __shared__ int histo_s[NUM_BINS];
    
    // Initialize private histogram
    for (int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    
    __syncthreads();
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid * CFACTOR; i < min((tid + 1) * CFACTOR, length); i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }
    
    // Merge phase (same as before)
    // ...
}
```

In this approach:
- Each thread processes a contiguous segment of `CFACTOR` elements
- Good for CPU execution (makes efficient use of cache lines)
- Suboptimal for GPU execution (doesn't maximize memory coalescing)

### Interleaved Partitioning

```cuda
__global__ void histogram_coarsening_interleaved(char* data, int length, int* histo) {
    __shared__ int histo_s[NUM_BINS];
    
    // Initialize private histogram
    for (int binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    
    __syncthreads();
    
    int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
        i += stride;
    }
    
    // Merge phase (same as before)
    // ...
}
```

Key advantages of interleaved partitioning:
- Threads in a warp access consecutive memory locations
- Enables memory coalescing for optimal memory bandwidth utilization
- Better suited for GPU architecture

## 9.6 Aggregation

When datasets contain localized concentrations of identical values, threads can experience heavy contention when updating the same histogram bins. Aggregation reduces this contention by:
- Tracking consecutive updates to the same bin
- Combining multiple increments into a single atomic operation
- Reducing the total number of atomic operations performed

```cuda
__global__ void histogram_aggregated_kernel(char* data, int length, int* histo) {
    __shared__ int histo_s[NUM_BINS];
    
    // Initialize private histogram
    // ...
    
    int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int accumulator = 0;
    int prevBinIdx = -1;
    
    while (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            int binIdx = alphabet_position/4;
            
            if (binIdx == prevBinIdx) {
                accumulator++;
            } else {
                if (prevBinIdx >= 0 && accumulator > 0) {
                    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
                }
                prevBinIdx = binIdx;
                accumulator = 1;
            }
        }
        i += stride;
    }
    
    // Flush any remaining aggregated updates
    if (prevBinIdx >= 0 && accumulator > 0) {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }
    
    // Merge phase (same as before)
    // ...
}
```

Implementation details:
- Each thread maintains an accumulator and tracks the current bin being updated
- When a different bin is encountered, the thread flushes accumulated updates
- A final flush is needed after processing all elements

Performance considerations:
- Most effective when data has localized concentrations of identical values
- Introduces overhead when contention is low (more variables and conditional logic)
- May reduce control divergence in highly contended scenarios

## 9.7 Summary

Parallel histogram computation represents an important class of parallel algorithms where:
- The output location for each thread is data-dependent
- Multiple threads may update the same output location
- Atomic operations are necessary to ensure correctness

Key challenges and solutions:
1. **Race Conditions**:
   - Read-modify-write operations can lead to lost updates
   - Atomic operations guarantee integrity but limit throughput
   - Atomic throughput is inversely proportional to memory latency

2. **Optimization Techniques**:
   - **Privatization**: Creates thread-local copies to reduce contention
   - **Shared Memory**: Reduces atomic operation latency
   - **Coarsening**: Reduces merge overhead by processing more elements per thread
   - **Interleaved Partitioning**: Maximizes memory coalescing on GPUs
   - **Aggregation**: Combines consecutive updates to the same bin

These techniques offer different trade-offs depending on:
- Hardware capabilities (atomic operation support in cache)
- Data distribution characteristics
- Available shared memory resources
- Required output precision

The optimization principles demonstrated through histogram computation are applicable to many other parallel algorithms facing output interference challenges.

