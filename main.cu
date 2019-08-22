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

class TestClass {

public:
    int * data;
    size_t len;



    TestClass(size_t len) {
        printf("Constructor\n");
        this->data = nullptr;
        this->len = len;
    }

    ~TestClass(){
        printf("~Destructor\n");
    }

    __host__ void allocate(){
        CUDACHECK(cudaMalloc((void**) &this->data, this->len * sizeof(int)));
        CUDACHECK(cudaMemset(this->data, 0, this->len * sizeof(int)));
    }

    __host__ void free(){
        CUDACHECK(cudaFree(this->data));
        this->data = nullptr;
    }

    __device__ int get(size_t index){
        return this->data[index];
    }
    __device__ void set(size_t index, int value){
        this->data[index] = value;
    }

};

__global__ void twodims_kernel(unsigned int maxx, unsigned int maxy){
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int gid = col + (row * (blockDim.x * gridDim.x));

    if(col < maxx && row < maxy){
        // Only print in some threads. 
        if (gid < 8){
            printf("gid %u, col: %u, row: %u valid\n", gid, col, row);
        }
        
    } else {
        if(gid < 8){
            printf("gid %u, col: %u, row: %u bad\n", gid, col, row);
        }
    }
}


void launch2dexample(){
    printf("launch2dexample\n");
    unsigned int XLEN = 8;
    unsigned int YLEN = 4;

    printf("problem size of %u x %u\n", XLEN, YLEN);

    unsigned int totalElements = XLEN * YLEN;
    
    // suggest block dimensions. Threads per block must not exceed 1024 on most hardware, registers will probably be a limiting factor. 
    dim3 blocksize(2, 2);

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
    twodims_kernel<<<blocksize, gridsize, 0, 0>>>(XLEN, YLEN);

    // synchronize after the kernel to make sure there were no errors. 
    CUDACHECK(cudaDeviceSynchronize());
    printf("launch2dexample finished\n");

}



__global__ void test_kernel(unsigned int threads, TestClass * d_instance){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < threads){
        // printf("Thread %u\n", tid);
        printf("Thread %u: d_isntance %p, element %d\n", tid, d_instance, d_instance->get(tid));
    }
}


void test_class_launch(){

    const size_t N = 16;

    // Construct on the host
    TestClass * h_instance = new TestClass(N);

    // Construct.
    printf("construct...\n");
    h_instance->allocate();

    printf("h_instance %p \n", h_instance);

    // Launch a kernel with the instance as the parameter

    printf("kernel...\n");
    test_kernel<<<N, 1>>>(N, h_instance);
    CUDACHECK(cudaDeviceSynchronize());
    printf("synced...\n");


    // Free
    printf("free...\n");
    h_instance->free();
    delete h_instance;

}
int main(int argc, char * argv[]){
    printf("main\n");

    launch2dexample();    

    return 1;
}
