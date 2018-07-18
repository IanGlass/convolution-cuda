/*
 * File: Conv_CPU.cpp
 * 
 * Author: Ian Glass
 * 
 * Date: 14/10/2013
 * 
 * Course: ENCE 463
 * 
 * Description: Module for performing Autocorrelation on the CPU using
 * convolution.
 * 
 * Usage: The module takes input data, width and height and returns 
 * the output matrix with the time taken.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>

// includes, cuda
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline cuDoubleComplex ComplexAdd(cuDoubleComplex a, cuDoubleComplex b)
{
    cuDoubleComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline cuDoubleComplex ComplexMul(cuDoubleComplex a, cuDoubleComplex b)
{
    cuDoubleComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Computes convolution on the host
void Convolve(const cuDoubleComplex *signal, const cuDoubleComplex *filter_kernel, int size, cuDoubleComplex *filtered_signal)
{
    int minRadius = size / 2;
    int maxRadius = size - minRadius;

    // Loop over output element indices
    for (int i = 0; i < size; ++i)
    {
        filtered_signal[i].x = filtered_signal[i].y = 0;

        // Loop over convolution indices
        for (int j = - maxRadius + 1; j <= minRadius; ++j)
        {
            int k = i + j;

            if (k >= 0 && k < size)
            {
                filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
extern "C" cuDoubleComplex *Conv_CPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time) {

	StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
	
	int width_out = width*2-1;
	int height_out = height*2-1;
	int size_padded = width_out*height_out;
	unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded; 
		
	//
    //create zero padded matrix before filling with input data
    cuDoubleComplex *data_padded = (cuDoubleComplex *) malloc(mem_size);
    for (int z = 0; z < size_padded; z++) {
		data_padded[z].x = 0;
		data_padded[z].y = 0;
	}
	
	//Fill matrix with input data and shift to center
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j ++) {
			data_padded[(j+((width-1)/2))+(i+((height-1)/2))*width_out].x = data[j+i*width].x;
			data_padded[(j+((width-1)/2))+(i+((height-1)/2))*width_out].y = data[j+i*width].y;
		}
	}
	
	//Allocate host memory for the convolution result
    cuDoubleComplex *conv_out = (cuDoubleComplex *)malloc(mem_size);
	//Convolve on the host
    Convolve(data_padded, data_padded, size_padded, conv_out);
    
    sdkStopTimer(&timer);
    *time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    //Tidy up
    free(data_padded);
    
    return(conv_out);
}






