#include <stdio.h>

__global__ void hello(){
    int thid=threadIdx.x;
    int blockId=blockIdx.x;
    printf("blockId=%d,thid=%d\n",blockId,thid);
}

int main() {
    hello<<<2,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
