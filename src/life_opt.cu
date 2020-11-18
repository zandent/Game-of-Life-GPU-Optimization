#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
#include "util.h"

extern "C" {
	#include "life_opt.h"
}
extern "C"
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

static double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000.0 + tv.tv_sec ;
}


/*****************************************************************************
 * Game of life implementation
 ****************************************************************************/
char* game_of_life_gpu (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max){
  
  debug_print("we're in game_of_life_gpu!\n");
  int a = 1;
  debug_print("we're in game_of_life_gpu %d!\n", a);
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
