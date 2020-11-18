#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
#include "util.h"

extern "C" {
	#include "life_opt.h"
}
extern "C"

static double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000.0 + tv.tv_sec ;
}

// 1: brute force implementation
// 2: bit implementation
#define GPU_IMPL_VERSION 1

////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// BIT IMPLEMENTATION, NO OPTIMIZATION ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
#if GPU_IMPL_VERSION == 2
#define BYTES_WINDOW 8
#define BYTES_PER_THREAD 1
#define LIVECHECK(count, state) (!state && (count == (char) 3)) ||(state && (count >= 2) && (count <= 3))
void ByteToBitCell(char* in, unsigned char* out, int row, int col, int colInBytes){
  memset(out,0,row*colInBytes*sizeof(char));
  for(int i = 0; i < row; i ++){
    for(int j = 0; j < colInBytes; j++){
      for(int k = 0; k < BYTES_WINDOW; k ++){
        //TODO: care about padding if remaining bits exist
        out[j+i*colInBytes] |= (in[k + j*BYTES_WINDOW + i*col] << (BYTES_WINDOW-k-1));
//        printf("%d ", in[k + j*BYTES_WINDOW + i*col]);
//        for (int ii = 0; ii < 8; ii++) {
//          printf("%d", !!((out[j+i*colInBytes] << ii) & 0x80));
//        }
//        printf("\n");
      }
//      printf("\n\n");
    }
  }
}
void BitCellToByte(unsigned char* in, char* out, int row, int col, int colInBytes){
  for(int i = 0; i < row; i ++){
    for(int j = 0; j < colInBytes; j++){
      for(int k = 0; k < BYTES_WINDOW; k ++){
        if((k + j*BYTES_WINDOW + i*col) < row*col){
          out[k + j*BYTES_WINDOW + i*col] = (in[j+i*colInBytes] >> (BYTES_WINDOW-k-1)) & 0x01;
        }
      }
    }
  }
}
__global__ void kernal(unsigned char* outboard, unsigned char* inboard, const int nrows, const int ncolsInBytes, const int noRemainBits){
	int ix = (threadIdx.x + blockIdx.x*blockDim.x)*BYTES_PER_THREAD;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
  if(ix<ncolsInBytes && iy<nrows){
    int lx = (ix+ncolsInBytes-1)%ncolsInBytes;
	  int uy = (iy+nrows-1)%nrows;
	  int dy = (iy+1)%nrows;
    uint row0 = (uint) inboard[lx+ncolsInBytes*uy] << 16;
    uint row1 = (uint) inboard[lx+ncolsInBytes*iy] << 16;
    uint row2 = (uint) inboard[lx+ncolsInBytes*dy] << 16;
    row0 |= (uint) inboard[ix+ncolsInBytes*uy] << 8;
    row1 |= (uint) inboard[ix+ncolsInBytes*iy] << 8;
    row2 |= (uint) inboard[ix+ncolsInBytes*dy] << 8;
    char result = 0x00;
    char count = 0x00;
    int base_x = ix;
	  int pre_x;
    for(int i = 0; i < BYTES_PER_THREAD; i++){
      if((base_x + i*BYTES_PER_THREAD) < ncolsInBytes){
        pre_x = ix;
        ix = (ix + 1)%ncolsInBytes;
        row0 |= (uint) inboard[ix+ncolsInBytes*uy];
        row1 |= (uint) inboard[ix+ncolsInBytes*iy];
        row2 |= (uint) inboard[ix+ncolsInBytes*dy];
        if(pre_x == ncolsInBytes-1){
          int mask = ~(0x01<<(noRemainBits + 7));
          row0 &= mask;
          row1 &= mask;
          row2 &= mask;
          row0 |= ((uint) inboard[ix+ncolsInBytes*uy] & 0x80) << noRemainBits;
          row1 |= ((uint) inboard[ix+ncolsInBytes*iy] & 0x80) << noRemainBits;
          row2 |= ((uint) inboard[ix+ncolsInBytes*dy] & 0x80) << noRemainBits;
        }
        result = 0x00;
        for(int j = 0; j < BYTES_WINDOW; j++){
          result <<= 1;
          count = ((row0 & 0x010000) >> 16) + ((row0 & 0x008000) >> 15) + ((row0 & 0x004000) >> 14) +
                  ((row1 & 0x010000) >> 16) +                             ((row1 & 0x004000) >> 14) +
                  ((row2 & 0x010000) >> 16) + ((row2 & 0x008000) >> 15) + ((row2 & 0x004000) >> 14);
          result |= LIVECHECK(count,(row1 & 0x008000));
          row0 <<= 1;
          row1 <<= 1;
          row2 <<= 1;
        }
        outboard[pre_x + ncolsInBytes*iy] = result;
      }
    }
  }
}

////////// Game of life implementation //////////
char* game_of_life_gpu (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max){
  debug_print("================== DEBUG MODE ===================\n");
  debug_print("we're in game_of_life_gpu! # iters = %d\n", gens_max);
  double timeStampA = getTimeStamp() ;

  int ncolsInBytes = ((ncols+BYTES_WINDOW-1)/BYTES_WINDOW);
  int noRemainBits = ncols%BYTES_WINDOW;
	int size = ncolsInBytes*nrows;
	int bytes = size*sizeof(char);
	unsigned char *d_bufA, *d_bufB;
  unsigned char parsed_inboard[size];
  unsigned char parsed_outboard[size];
  ByteToBitCell(inboard, parsed_inboard, nrows, ncols, ncolsInBytes);
//  for (int ii = 0; ii < nrows; ii++){ 
//    for (int jj = 0; jj < ncols; jj++){ 
//      printf ("%d", inboard[jj+ii*ncols]);
//    }
//    printf("\n");
//  }
//  printf("\n");
//  for (int ii = 0; ii < nrows; ii++){ 
//    for (int jj = 0; jj < ncolsInBytes; jj++){ 
//      //printf ("%x", parsed_inboard[jj+ii*ncols]&0xff);
//      for (int i = 0; i < 8; i++) {
//        printf("%d", !!((parsed_inboard[jj+ii*ncolsInBytes] << i) & 0x80));
//      }
//    }
//    printf("\n");
//  }
	cudaMalloc((void **)&d_bufA,bytes);
	cudaMalloc((void **)&d_bufB,bytes);
	cudaMemcpy( d_bufA, parsed_inboard, bytes, cudaMemcpyHostToDevice);
	dim3 block(32,32);
	dim3 grid(((ncolsInBytes+BYTES_PER_THREAD-1)/BYTES_PER_THREAD+block.x-1)/block.x,(nrows + block.y-1)/block.y);
	for (int curgen = 0; curgen < gens_max; curgen++){
		kernal<<<grid,block>>>(d_bufB, d_bufA, nrows, ncolsInBytes, noRemainBits);
    cudaDeviceSynchronize() ;
    //SWAP BOARDS
    unsigned char * temp = d_bufA;
    d_bufA = d_bufB;
		d_bufB = temp;
	}
	cudaMemcpy(parsed_outboard, d_bufA, bytes, cudaMemcpyDeviceToHost);
    BitCellToByte(parsed_outboard, outboard, nrows, ncols, ncolsInBytes);
    
  double timeStampD = getTimeStamp() ;
	double total_time = timeStampD - timeStampA;
	printf("GPU game_of_life: %.6f\n", total_time);
	return outboard;
}
#endif //GPU_IMPL_VERSION == 2


////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////...//////// BRUTE FORCE, NO OPTIMIZATION ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
#if GPU_IMPL_VERSION == 1
#define LIVECHECK(count, state) (!state && (count == (char) 3)) ||(state && (count >= 2) && (count <= 3))
__global__ void kernal(char* outboard, char* inboard, const int nrows, const int ncols, const int size){
  int ix = threadIdx.x + blockIdx.x*blockDim.x;
  int iy = threadIdx.y + blockIdx.y*blockDim.y;
  int index = ix + blockDim.x*gridDim.x*iy;
  if(ix<ncols && iy<nrows){
    int rx = (ix+1)%ncols;
    int lx = (ix+ncols-1)%ncols;
    int uy = (iy+nrows-1)%nrows;
    int dy = (iy+1)%nrows;
    char state = inboard[index];
    char count = inboard[lx+ncols*uy] + inboard[ix+ncols*uy] + inboard[rx+ncols*uy] + inboard[lx+ncols*iy] + inboard[rx+ncols*iy] + inboard[lx+ncols*dy] + inboard[ix+ncols*dy] + inboard[rx+ncols*dy];
   outboard[index] = LIVECHECK(count,state);
  }
}

////////// Game of life implementation //////////

char* game_of_life_gpu (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max){
  debug_print("================== DEBUG MODE ===================\n");
  debug_print("we're in game_of_life_gpu! # iters = %d\n", gens_max);
  double timeStampA = getTimeStamp() ;
  int size = ncols*nrows;
  int bytes = size*sizeof(char);
  char *d_bufA, *d_bufB;
  cudaMalloc((void **)&d_bufA,bytes);
  cudaMalloc((void **)&d_bufB,bytes);
  cudaMemcpy( d_bufA, inboard, bytes, cudaMemcpyHostToDevice);
  //cudaMemcpy( d_bufB, outboard, bytes, cudaMemcpyHostToDevice);
  dim3 block(32,32);
  dim3 grid((ncols+block.x-1)/block.x,(nrows + block.y-1)/block.y);
  for (int curgen = 0; curgen < gens_max; curgen++){
    kernal<<<grid,block>>>(d_bufB, d_bufA, nrows, ncols, size);
    //SWAP BOARDS
    char * temp = d_bufA;
    d_bufA = d_bufB;
    d_bufB = temp;
  }
  cudaMemcpy(outboard, d_bufA, bytes, cudaMemcpyDeviceToHost);

  double timeStampD = getTimeStamp() ;
  double total_time = timeStampD - timeStampA;
  printf("GPU game_of_life: %.6f\n", total_time);
  return outboard;
}
#endif //GPU_IMPL_VERSION == 1














