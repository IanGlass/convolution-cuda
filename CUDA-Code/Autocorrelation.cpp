/*
 * File: Autocorrelation.cpp
 * 
 * Author: Ian Glass
 * 
 * Date: 14/10/2013
 * 
 * Description: Main module for performing autocorrelation. 
 * Equipped to perform multiple calls of convolution on the CPU, GPU and
 * a Fast Fourier Transform on the GPU.
 * 
 * Usage: Making call to run_program will execute all three forms of 
 * autocorrelation, taking width, height and input data as arguments to 
 * name a few.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>

//includes, cuda
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

int *pArgc = NULL;
char **pArgv = NULL;

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" cuDoubleComplex *FFT_Shift_GPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time);
extern "C" cuDoubleComplex *FFT_Rot_GPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time);
extern "C" cuDoubleComplex *Conv_GPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time);
extern "C" cuDoubleComplex *Conv_CPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void run_programs(int argc, char **argv, int Width, int Height, cuDoubleComplex *data, 
cuDoubleComplex *Result_convGPU, cuDoubleComplex *Result_convCPU, cuDoubleComplex *Result_FFTshift) {
	//calculate required dimensions of output results
	int width_out = Width*2-1;
	int height_out = Height*2-1;
	//calculate size after padding (important for rest of code)
	int size_padded = width_out*height_out;
	unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded;
	
	//Temporary storage for time values
	float *Time = (float*)malloc(sizeof(float));
	
	//Floats for comparing output matrices
	float *convCPU = (float *)malloc(sizeof(float)*width_out*height_out);
	float *convGPU = (float *)malloc(sizeof(float)*width_out*height_out);
	float *FFTshift = (float *)malloc(sizeof(float)*width_out*height_out);
	//
	
	//	
	//Convolution on CPU
	Result_convCPU = Conv_CPU_fn(argc, argv, data, Width, Height, Time);
	//Load results into floats * for direct comparison
	for (int a = 0; a < height_out; a++) {
		for (int b = 0; b < width_out; b++) {
			convCPU[b+a*width_out] = (float)Result_convCPU[b+a*width_out].x;
		}
	}
	printf("\nProcessing time: %f (ms) CPU",*Time);
	
	//Convolution on GPU
	Result_convGPU = Conv_GPU_fn(argc, argv, data, Width, Height, Time);
	//Load results into floats * for direct comparison
	for (int c = 0; c < height_out; c++) {
		for (int d = 0; d < width_out; d++) {
			convGPU[d+c*width_out] = (float)Result_convGPU[d+c*width_out].x;
		}
	}
	//Perform comparison
	bool convGPUCorrect = sdkCompareL2fe((float *)convGPU, (float *)convCPU, size_padded, 1e-5f);
	printf("\nProcessing time: %f (ms) GPU",*Time);
    
    //FFT with shift on GPU
    Result_FFTshift = FFT_Shift_GPU_fn(argc, argv, data, Width, Height, Time);
    //Load results into floats * for direct comparison
    for (int e = 0; e < height_out; e++) {
		for (int f = 0; f < width_out; f++) {
			FFTshift[f+e*width_out] = (float)Result_FFTshift[f+e*width_out].x;
		}
	}
	//Perform comparison
	bool FFTshiftCorrect = sdkCompareL2fe((float *)FFTshift, (float *)convCPU, size_padded, 1e-5f);
    printf("\nProcessing time: %f (ms) FFT",*Time);
    //
    //	
    
    //Print comparison results
    if (convGPUCorrect) {
		printf("\nGPU convolution match");
	}
	if (FFTshiftCorrect) {
		printf("\nFFT shift match");
	}
	//
	//
	
	free(convCPU);
	free(convGPU);
	free(FFTshift);
	free(Time);
}

int main(int argc, char **argv) {
	pArgc = &argc;
    pArgv = argv;
    
    int width = 15;
    int height = 15;
    
    cuDoubleComplex *matrix = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*width*height);
	
	//Initialize test matrix
	for (int i = 0; i < (width*height); i++) {
		matrix[i].x = 1;
		matrix[i].y = 0;
	}	
	
	//Calculate size required for output matrix, based on padding	
	int width_out = width*2-1;
	int height_out = height*2-1;
	int size_padded = width_out*height_out;
	unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded;
	
	//Allocate storage for output matrices
	cuDoubleComplex *CPU_Conv = (cuDoubleComplex *)malloc(mem_size);
	cuDoubleComplex *GPU_Conv = (cuDoubleComplex *)malloc(mem_size);
	cuDoubleComplex *GPU_FFTshift = (cuDoubleComplex *)malloc(mem_size);
	
	//Execute all three implementations
	run_programs(argc, argv, width, height, matrix, CPU_Conv, GPU_Conv, GPU_FFTshift);
			
	free(matrix);
	free(CPU_Conv);
	free(GPU_Conv);
	free(GPU_FFTshift);
    
    printf("\n");
    exit(EXIT_SUCCESS);
}







