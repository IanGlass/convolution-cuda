/*
 * File: FFT_Shift_GPU_kernel.cpp
 * 
 * Author: Ian Glass
 * 
 * Date: 14/10/2013
 * 
 * Course: ENCE 463
 * 
 * Description: Module for performing Autocorrelation on the GPU using
 * FFT.
 * 
 * Usage: The module takes input data, width and height and returns 
 * the output matrix with the time taken.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h> // helper functions for SDK examples

typedef unsigned int  uint;
typedef unsigned char uchar;

//Defines the maximum number of threads per block
#define thread_limit 512

////////////////////////////////////////////////////////////////////////////////
//Multiply by complex conjugate
__global__ void Matrix_Multiply(cuDoubleComplex *Mul_in, cuDoubleComplex *Mul_out) {
	
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
		
	Mul_out[ID].x = Mul_in[ID].x*Mul_in[ID].x+Mul_in[ID].y*Mul_in[ID].y;
	//Complex output always zero due to maths
	Mul_out[ID].y = 0;

	__syncthreads();
}

//Zero-pads the input matrix, not shifted as performed in convolution
__global__ void Pad_FFT(cuDoubleComplex *input, cuDoubleComplex *output, int width, int width_out, int height) {
	
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Calculate row and column position for filling data
	int j = ID%width_out;
	int i = ID/width_out;
	//
	
	output[ID].x = 0;
	output[ID].y = 0;
	
	//Fill matrix with input data
	if ((j<width)&&(i<height)) {
		output[j+i*width_out].x = input[j+i*width].x;
		//Omit complex component as it is set to zero in Multiply
	}
	//
	
	__syncthreads();
}

//Shifts IFFT data back to correct places
__global__ void Shift(cuDoubleComplex *input, cuDoubleComplex *output, int width, int width_out, int height, int height_out, int size) {
	
	//Find current position in matrix as 1D index
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ID <= size) {
		//Calculate row and column position for filling data
		int j = ID%width_out;
		int i = ID/width_out;
		//
		
		//Top left output section
		if ((j >= width) && (i >= (height-1))) {
			output[(j-width)+(i-height+1)*width_out].x = input[j+i*width_out].x/size;
		}
		//
		
		//Bottom left output section
		else if ((j >= width) && (i < height)) {
			output[(j-width)+(height+i)*width_out].x = input[j+i*width_out].x/size;
		}
		//
		
		//Top right output section
		else if ((j < width) && (i >= height)) {
			output[(width+j-1)+(i-height)*width_out].x = input[j+i*width_out].x/size;
		}
		//
		
		//Bottom right output section
		else if ((j < width) && (i < height)) {
			output[(width+j-1)+(height+i-1)*width_out].x = input[j+i*width_out].x/size;
		}
		//
		
	}
	
	__syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Module main
////////////////////////////////////////////////////////////////////////////////
extern "C" cuDoubleComplex *FFT_Shift_GPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time) {
	
	int width_out = width*2-1;
	int height_out = height*2-1;
	 
    printf("\n%s Starting...", argv[0]);
    
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);
    
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    int size_padded = width_out*height_out;
    
    //Set thread conditions
    int num_threads = thread_limit;
    if (size_padded < thread_limit) {
		num_threads = size_padded;
	}
	//
	
    unsigned int mem_size_FFT = sizeof(cuDoubleComplex)*size_padded;       
    
    //setup execution parameters
    //calculate required grid dimensions
    int grid_size = size_padded/thread_limit+1;
    dim3 grid(grid_size, 1, 1);
    dim3 threads(num_threads, 1, 1);
    printf("\nInit grid %f", sdkGetTimerValue(&timer));
    //
    
    //Move input data to device
    cuDoubleComplex *data_in;
    checkCudaErrors(cudaMalloc((void **)&data_in, sizeof(cuDoubleComplex)*width*height));
    checkCudaErrors(cudaMemcpy(data_in, data, sizeof(cuDoubleComplex)*width*height, cudaMemcpyHostToDevice));
    printf("\nMem copied to device %f", sdkGetTimerValue(&timer));
    //
    
    //create zero padded matrix
    cuDoubleComplex *data_padded;
    checkCudaErrors(cudaMalloc((void **)&data_padded, mem_size_FFT));
    Pad_FFT<<<grid,threads>>>(data_in,data_padded, width, width_out, height);
    printf("\nZero-padded %f", sdkGetTimerValue(&timer));
    //
    
    //Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, width_out*height_out, CUFFT_Z2Z, 1);
    //
    
    //Perform FFT
    cuDoubleComplex *data_FFT;
    checkCudaErrors(cudaMalloc((void **)&data_FFT, mem_size_FFT));
    checkCudaErrors(cufftExecZ2Z(plan, data_padded, data_FFT, CUFFT_FORWARD));
    cudaThreadSynchronize();
    printf("\nFFT performed %f", sdkGetTimerValue(&timer));
    //
    
    //Multiply by conjugate
    cuDoubleComplex *data_Multiply;
    checkCudaErrors(cudaMalloc((void **)&data_Multiply, mem_size_FFT));
	Matrix_Multiply<<<grid,threads>>>(data_FFT, data_Multiply);
	printf("\nMultiplied %f", sdkGetTimerValue(&timer));
	//
	    
    //Perform IFFT
    checkCudaErrors(cufftExecZ2Z(plan, data_Multiply, data_Multiply, CUFFT_INVERSE));
    cudaThreadSynchronize();
    printf("\nIFFT performed %f", sdkGetTimerValue(&timer));
    //
    
    //Perform shift
    cuDoubleComplex *Shifted;
    checkCudaErrors(cudaMalloc((void **)&Shifted, mem_size_FFT));
    Shift<<<grid,threads>>>(data_Multiply, Shifted, width, width_out, height, height_out, size_padded);
    printf("\nShifted %f", sdkGetTimerValue(&timer));
    //
    
    //Move result to host
    cuDoubleComplex *Correlated = (cuDoubleComplex *)malloc(mem_size_FFT);
    checkCudaErrors(cudaMemcpy(Correlated, Shifted, mem_size_FFT, cudaMemcpyDeviceToHost));
    printf("\nMem copied to host %f", sdkGetTimerValue(&timer));
    //
                  
    sdkStopTimer(&timer);
    *time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    //Tidy up
    cufftDestroy(plan);
    cudaFree(data_in);
    cudaFree(data_padded);
    cudaFree(data_FFT);
    cudaFree(data_Multiply);
    cudaFree(Shifted);
    
    cudaDeviceReset();
    
    return(Correlated);
}



