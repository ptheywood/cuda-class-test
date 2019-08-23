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
    
    
    // Get the number of threads to launch 

    // Ask the occupancy calculator to find the total number of threads per block which will maximiuse occupancy for the kernel.
    int minGridSize = 0; // Minimum grid size to achieve max occupancy
    int totalThreadsPerBlock = 0; // Number of threads per block
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &totalThreadsPerBlock, twodims_kernel, 0, 0));


    // Given the calculated blocksize, figure out each dimension for some form of 2D grid.
    // This could be non square to fit the same shape as the b.,problem, but we will assume square for now
    int ttpb_sqrt = (int)floor(sqrt(totalThreadsPerBlock));

    printf("mgs: %d, tpb %d, sqrt %d\n", minGridSize, totalThreadsPerBlock, ttpb_sqrt);


    // suggest block dimensions. Threads per block must not exceed 1024 on most hardware, registers will probably be a limiting factor. 
    dim3 blocksize(ttpb_sqrt, ttpb_sqrt);

    // shrink either if larger than the actual dimensions to minimise work
    if(blocksize.x > XLEN){
        blocksize.x = XLEN;
    }
    if(blocksize.y > YLEN){
        blocksize.y = YLEN;
    }

    dim3 gridsize;
    gridsize.x = (XLEN + blocksize.x - 1) / blocksize.x;
    gridsize.y = (YLEN + blocksize.y - 1) / blocksize.y;


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
