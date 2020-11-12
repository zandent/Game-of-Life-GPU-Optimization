#include <cuda_runtime.h>
extern "C" {
	#include "life_opt.h"
}
extern "C"
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

/*****************************************************************************
 * Game of life implementation
 ****************************************************************************/
char* game_of_life (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max){
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
	return outboard;
}
