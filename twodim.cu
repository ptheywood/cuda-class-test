#include <stdio.h>
#include <cuda_runtime.h>

#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template<typename T>
inline void
get_kernel_dims(const int max_x,
                const int max_y,
                T kernel,
                dim3& out_blocksize,
                dim3& out_gridsize)
{

    // Use the occupancy calculator to find the 1D numbr of threads per block which maximises occupancy. Assumes a square number. 
    int minGridSize = 0; // Minimum grid size to achieve max occupancy
    int totalThreadsPerBlock = 0; // Number of threads per block
    // Query the occupancy calculator.
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &totalThreadsPerBlock, kernel, 0, 0));

    // Assume we alwasy want square kernels. This may be sub-optimal.
    int blocksize_xy = (int)floor(sqrt(totalThreadsPerBlock));


    // Suggest block dimensions. Threads per block must not exceed 1024 on most
    // hardware, registers will probably be a limiting factor.
    dim3 blocksize(blocksize_xy, blocksize_xy);

    // Shrink either if larger than the actual dimensions to minimise work
    // @note this might reduce the work below ideal occupancy, for very wide/narrow problems
    if (blocksize.x > max_x) {
    blocksize.x = max_y;
    }
    if (blocksize.y > max_x) {
    blocksize.y = max_y;
    }

    // Calculate the gridsize. 
    dim3 gridsize;
    gridsize.x = (max_x + blocksize.x - 1) / blocksize.x;
    gridsize.y = (max_y + blocksize.y - 1) / blocksize.y;

    //  Set for the outside ones. 
    out_blocksize = blocksize;
    out_gridsize = gridsize;
}
 
__global__ void twodims_kernel(unsigned int maxx, unsigned int maxy){
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int gid = col + (row * (blockDim.x * gridDim.x));

    if(col < maxx && row < maxy){
        // Only print in some threads. 
        if (gid < 8){
            //printf("gid %u, col: %u, row: %u valid\n", gid, col, row);
        }
        
    } else {
        if(gid < 8){
            //printf("gid %u, col: %u, row: %u bad\n", gid, col, row);
        }
    }
}


void launch2dexample(){
    printf("launch2dexample\n");
    unsigned int XLEN = 256;
    unsigned int YLEN = 768;

    printf("problem size of %u x %u\n", XLEN, YLEN);

    unsigned int totalElements = XLEN * YLEN;
    
    dim3 blocksize;
    dim3 gridsize;

    get_kernel_dims(XLEN, YLEN, twodims_kernel, blocksize, gridsize);


    // Given the calculated blocksize, figure out each dimension for some form of 2D grid.
    // This could be non square to fit the same shape as the b.,problem, but we will assume square for now

    unsigned int totalThreads = (blocksize.x * blocksize.y) * (gridsize.x * gridsize.y);

    printf("Launching %d x %d threads per block, with %d x %d blocks.\n %u elements, %u threads\n",
        blocksize.x, blocksize.y, gridsize.x, gridsize.y, totalElements, totalThreads);

    // Launch the kernel. 
    twodims_kernel<<<gridsize, blocksize, 0, 0>>>(XLEN, YLEN);

    // synchronize after the kernel to make sure there were no errors. 
    CUDACHECK(cudaDeviceSynchronize());
    printf("launch2dexample finished\n");

}

int main(int argc, char * argv[]){
    printf("main\n");

    launch2dexample();    

    return 1;
}
