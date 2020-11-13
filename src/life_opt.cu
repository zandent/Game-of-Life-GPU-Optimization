#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>

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

static double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000.0 + tv.tv_sec ;
}


/*****************************************************************************
 * Game of life implementation
 ****************************************************************************/
char* game_of_life_gpu (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max){
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

///// EXAMPLE IMPLEMENTATION. 
/*
__global__ void bitLifeKernelNoLookup(const ubyte* lifeData, uint worldDataWidth,
    uint worldHeight, uint bytesPerThread, ubyte* resultLifeData) 
{
 
	uint worldSize = (worldDataWidth * worldHeight);

	for (uint cellId = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * bytesPerThread;
	  	 cellId < worldSize;
	     cellId += blockDim.x * gridDim.x * bytesPerThread) 
	{
		uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
		uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
		uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
		uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

		// Initialize data with previous byte and current byte.
		uint data0 = (uint)lifeData[x + yAbsUp] << 16;
		uint data1 = (uint)lifeData[x + yAbs] << 16;
		uint data2 = (uint)lifeData[x + yAbsDown] << 16;

		x = (x + 1) % worldDataWidth;
		data0 |= (uint)lifeData[x + yAbsUp] << 8;
		data1 |= (uint)lifeData[x + yAbs] << 8;
		data2 |= (uint)lifeData[x + yAbsDown] << 8;

		for (uint i = 0; i < bytesPerThread; ++i) 
		{
			uint oldX = x;  // old x is referring to current center cell
			x = (x + 1) % worldDataWidth;
			data0 |= (uint)lifeData[x + yAbsUp];
			data1 |= (uint)lifeData[x + yAbs];
			data2 |= (uint)lifeData[x + yAbsDown];

			uint result = 0;
			for (uint j = 0; j < 8; ++j) 
			{
				uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
				aliveCells >>= 14;
				aliveCells = (aliveCells & 0x3) + (aliveCells >> 2)
				  + ((data0 >> 15) & 0x1u) + ((data2 >> 15) & 0x1u);

				result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

				data0 <<= 1;
				data1 <<= 1;
				data2 <<= 1;
			}

			resultLifeData[oldX + yAbs] = result;
		}
	}
}

void runSimpleLifeKernel(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer, size_t worldWidth,
    size_t worldHeight, size_t iterationsCount, ushort threadsCount) 
{
	assert((worldWidth * worldHeight) % threadsCount == 0);
	size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
	ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

	for (size_t i = 0; i < iterationsCount; ++i) 
	{
		simpleLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, worldWidth, worldHeight, d_lifeDataBuffer);
		std::swap(d_lifeData, d_lifeDataBuffer);
	}
}
*/
