#include <stdio.h>
#include <cuda_runtime.h>

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
        cudaMalloc((void**) &this->data, this->len * sizeof(int));
        cudaMemset(this->data, 0, this->len * sizeof(int));
    }

    __host__ void free(){
        cudaFree(this->data);
        this->data = nullptr;
    }

    __device__ int get(size_t index){
        return this->data[index];
    }
    __device__ void set(size_t index, int value){
        this->data[index] = value;
    }

};


__global__ void test_kernel(unsigned int threads, TestClass * d_instance){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < threads){
        // printf("Thread %u\n", tid);
        printf("Thread %u: d_isntance %p, element %d\n", tid, d_instance, d_instance->get(tid));
    }
}


int main(int argc, char * argv[]){
    printf("main\n");

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
    cudaDeviceSynchronize();
    printf("synced...\n");


    // Free
    printf("free...\n");
    h_instance->free();
    delete h_instance;

    return 1;
}
