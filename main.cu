#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <thrust/sort.h>
using namespace cooperative_groups;

 struct     pairs{
    char4 *data;
    unsigned int*length;

    __device__ __host__
    int cmp4(const char4 & c1, const char4 & c2)const
    {
        int result;

        result = c1.x - c2.x; if (result !=0) return result;
        result = c1.y - c2.y; if (result !=0) return result;
        result = c1.z - c2.z; if (result !=0) return result;
        result = c1.w - c2.w; if (result !=0) return result;

        return 0;
    }

    __device__ __host__
    int strncmp4(const char4 * s1, const char4 * s2, const unsigned int nwords)const
    {
        for(unsigned int i=0; i<nwords; i++) {
            int result = cmp4(s1[i], s2[i]);
            if (result != 0) return result;
        }

        return 0;
    }
    __device__ __host__  bool operator==(const pairs b)const {
        unsigned int len=min(*(this->length),*(b.length));
        return (strncmp4(this->data,b.data,len)==0);
    }

     __device__ __host__  bool operator!=(const pairs b)const {
         unsigned int len=min(*(this->length),*(b.length));
         return (strncmp4(this->data,b.data,len)!=0);
     }
     __device__ __host__  bool operator<(const pairs b)const {
         unsigned int len=min(*(this->length),*(b.length));
         return (strncmp4(this->data,b.data,len)<0);
     }
     __device__ __host__  bool operator<=(const pairs b)const{
         unsigned int len=min(*(this->length),*(b.length));
         return (strncmp4(this->data,b.data,len)<=0);
     }
     __device__ __host__  bool operator>(const pairs b)const{
         unsigned int len=min(*(this->length),*(b.length));
         return (strncmp4(this->data,b.data,len)>0);
     }
     __device__ __host__  bool operator>=(const pairs b)const{
         unsigned int len=min(*(this->length),*(b.length));
         return (strncmp4(this->data,b.data,len)>=0);
     }
};


__device__ int reduce(thread_group&g,int*x,int val){
    int lane=g.thread_rank();
//    printf("thread_rank=%d\n",lane);

    for(int i=g.size()/2;i>0;i=i/2){
        x[lane]=val;g.sync();
        val=val+x[lane+i];g.sync();
        if(lane==31){
            printf("val=%d\n",val);
        }
    }
//    printf("lane=%d,val=%d\n",lane,val);
    return val;
}
__global__ void parallel_kenel(int *x){
    int thid=threadIdx.x+blockIdx.x*blockDim.x;
    thread_block my_block = this_thread_block();//this block
//    printf("thread_rank=%d\n",my_block.thread_rank());
    auto my_tile=tiled_partition<32>(my_block);//partition 32 threads in one block;
    int result=reduce(my_tile,x+32*blockIdx.x,1);
//    printf("result=%d\n",result);

}
/**
 *
 * @param key insert key
 * @param bloom bloom filter
 * @param table hash table
 */
__global__ void insert_bloom(int*key,int*bloom,int*table){
    int thid=threadIdx.x+blockIdx.x*blockDim.x;
    int insert_key=key[thid];
    unsigned hash=(insert_key>>25)<<3;//
    //取余数
    unsigned value=0;
    for(int i=0;i<8;i++){//使用8个hash函数
        int sector_id=hash+i;
        value=(insert_key*table[i])&0x1f;//32bit
        atomicOr(bloom+sector_id,(1<<value));
    }
}
template<unsigned size>
__device__ bool search_bloom(thread_block_tile<size> g,int*key,int*bloom,int *table){
    int thid=threadIdx.x+blockIdx.x*blockDim.x;
    int lane=g.thread_rank();
    unsigned keys=key[thid/8];
    unsigned hash=(keys>>25)<<3;
    unsigned value=(keys*table[lane])&&0x1f;
    //query
    unsigned query_value=bloom[hash+lane]&(1<<value);
    unsigned tmp=g.ballot(query_value==0);
    return tmp==0;
}

__global__ void parallel_search(int*key,int*bloom,int*table){
    int thid=threadIdx.x+blockIdx.x*blockDim.x;
    thread_block my_block = this_thread_block();//this block
    auto my_tile=tiled_partition<32>(my_block);//partition 32 threads in one block;
    bool result=search_bloom(my_tile,key,bloom,table);
    if(thid%8==0){
        if(result==1)
            printf("{%d,%d}",key[thid/8],result);
    }
}

__global__  void showDeviceSort(pairs*p){
    printf("data=%c\n",p[0].data[0].x);
}
int main(){
    int test_len=3;
    pairs*h_p,*d_p,*tmp_p;//中间变量 临时指针用于分配空间
    h_p=(pairs*)malloc(test_len*sizeof(pairs));
    tmp_p=(pairs*)malloc(test_len*sizeof (pairs));
    //data 0
    h_p[0].length=(unsigned int*)malloc(sizeof(unsigned int));
    h_p[0].data=(char4*)malloc(sizeof(char4));
    h_p[0].length[0]=4;
    h_p[0].data[0].x='z';
    h_p[0].data[1].y='b';
    h_p[0].data[2].z='c';
    h_p[0].data[3].w='d';
    cudaMalloc(&(tmp_p[0].length),sizeof (unsigned int));
    cudaMalloc(&(tmp_p[0].data),sizeof(char4));
    cudaMemcpy(tmp_p[0].length,h_p[0].length,sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_p[0].data,h_p[0].data,sizeof(char4),cudaMemcpyHostToDevice);
    //data 1
    h_p[1].length=(unsigned int*)malloc(sizeof(unsigned int));
    h_p[1].data=(char4*)malloc(sizeof(char4));
    h_p[1].length[0]=4;
    h_p[1].data[0].x='e';
    h_p[1].data[1].y='b';
    h_p[1].data[2].z='c';
    h_p[1].data[3].w='d';
    cudaMalloc(&(tmp_p[1].length),sizeof (unsigned int));
    cudaMalloc(&(tmp_p[1].data),sizeof(char4));
    cudaMemcpy(tmp_p[1].length,h_p[1].length,sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_p[1].data,h_p[1].data,sizeof(char4),cudaMemcpyHostToDevice);
    //data 2
    h_p[2].length=(unsigned int*)malloc(sizeof(unsigned int));
    h_p[2].data=(char4*)malloc(sizeof(char4));
    h_p[2].length[0]=4;
    h_p[2].data[0].x='a';
    h_p[2].data[1].y='b';
    h_p[2].data[2].z='c';
    h_p[2].data[3].w='d';
    cudaMalloc(&(tmp_p[2].length),sizeof (unsigned int));
    cudaMalloc(&(tmp_p[2].data),sizeof(char4));
    cudaMemcpy(tmp_p[2].length,h_p[2].length,sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_p[2].data,h_p[2].data,sizeof(char4),cudaMemcpyHostToDevice);
    cudaMalloc(&d_p,test_len*sizeof(pairs));
    cudaMemcpy(d_p,tmp_p,test_len*sizeof(pairs),cudaMemcpyHostToDevice);
    thrust::sort(thrust::device,d_p,d_p+3);
    showDeviceSort<<<1,1>>>(d_p);
    cudaDeviceSynchronize();
    printf("%c\n",h_p[0].data[0].x);
    return 0;
}
int mains() {
    //test cooperative code
//    int *h_x,*d_x;
//    h_x=(int*)malloc(1024*sizeof(int));
//    memset(h_x,1,1024*sizeof(int));
//    cudaMalloc(&d_x,1024*sizeof(int));
//    cudaMemcpy(d_x,h_x,1024*sizeof(int),cudaMemcpyHostToDevice);
//    parallel_kenel<<<3,32>>>(d_x);
//    cudaDeviceSynchronize();
//    return 0;
    //test code end
    srand((unsigned int)time(0));//随机种子
    int insert_size=(1<<20);
    int*h_key,*d_key;
    h_key=(int*)malloc(insert_size*sizeof(int));
    cudaMalloc(&d_key,insert_size*sizeof(int));
    for(int i=0;i<insert_size;i++){
        h_key[i]=rand()%(1<<31);
    }
    cudaMemcpy(d_key,h_key,insert_size*sizeof(int),cudaMemcpyHostToDevice);

    //lookup key
    int query_size=(1<<15);
    int*h_query,*d_query;
    h_query=(int*)malloc(query_size*sizeof(int));
    cudaMalloc(&d_query,query_size*sizeof(int));
    for(int i=0;i<query_size;i++){
        h_query[i]=rand()%(1<<31);
    }
    cudaMemcpy(d_query,h_query,query_size*sizeof(int),cudaMemcpyHostToDevice);

    //bloom filter,int*;
    int bloom_size=(1<<10);//sector length
    //block length=sector length/32;
    int*h_bloom,*d_bloom;
    h_bloom=(int*)malloc(bloom_size*sizeof(int));
    cudaMalloc(&d_bloom,bloom_size*sizeof(int));
    memset(h_bloom,0,bloom_size*sizeof(int));
    cudaMemcpy(d_bloom,h_bloom,bloom_size*sizeof(int),cudaMemcpyHostToDevice);

    //create hash table
    int*d_table;
    cudaMalloc(&d_table,32*sizeof (int));
    int h_table[32]={3 ,5, 7 ,11 ,13 ,17 ,19 ,23 ,29 ,31, 37, 41, 43 ,47, 53,59,
                     61,67,71 ,73,79 ,83 ,89, 97,101,103,107,109,113,127,131,137};
    cudaMemcpy(d_table,h_table,32*sizeof(int),cudaMemcpyHostToDevice);

    insert_bloom<<<1024,1024>>>(d_key,d_bloom,d_table);
    cudaDeviceSynchronize();
    parallel_search<<<1024,256>>>(d_query,d_bloom,d_table);
    cudaDeviceSynchronize();
    return 0;
}
