/**************************************************************************//**
* @file Conv_GPU_kernel.cpp
* @brief Module for performing Autocorrelation on the GPU using 
* convolution. The module takes input data, width and height and returns 
* the output matrix with the time taken.
* @author Ian Glass
* @date    14/10/2013
*******************************************************************************
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*******************************************************************************/

/* Includes ------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h> // helper functions for SDK examples

/* Global Variables ----------------------------------------------------------*/
typedef unsigned int  uint;
typedef unsigned char uchar;

/* Defines the maximum number of threads per block */
#define thread_limit 512

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

/*-----------------------------------------------------------*/
/**
  * @brief Addition of complex values on the GPU.
  * @param  Two complex variables to be added.
  * @retval None
  */
static __device__ __host__ inline cuDoubleComplex ComplexAdd(cuDoubleComplex a, cuDoubleComplex b) {
    cuDoubleComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

/*-----------------------------------------------------------*/
/**
  * @brief Multiplication of complex values on the GPU.
  * @param  Two complex variables to be added.
  * @retval None
  */
static __device__ __host__ inline cuDoubleComplex ComplexMul(cuDoubleComplex a, cuDoubleComplex b) {
    cuDoubleComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

/*-----------------------------------------------------------*/
/**
  * @brief Computes convolution on the GPU.
  * @param  Two complex variables to be added.
  * @retval None
  */
__global__ void Convolve(cuDoubleComplex *signal, cuDoubleComplex *filter_kernel, int size, cuDoubleComplex *filtered_signal) {
    int minRadius = size / 2;
    int maxRadius = size - minRadius;
	
	/* Find current position in matrix as 1D index */
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	
    /* Loop over output element indices */
    filtered_signal[ID].x = filtered_signal[ID].y = 0;

    /* Loop over convolution indices */
    for (int j = - maxRadius + 1; j <= minRadius; ++j) {
		int k = ID + j;

        if (k >= 0 && k < size) {
			filtered_signal[ID] = ComplexAdd(filtered_signal[ID], ComplexMul(signal[k], filter_kernel[minRadius - j]));
        }
    }
}

/*-----------------------------------------------------------*/
/**
  * @brief Zero-pads the input matrix and shifts it by (width-1)/2
  * @param  The input matrix, a pointer to the result, the width of the
  * input matrix, the width of the output matrix and the height of the 
  * input matrix.
  * @retval None
  */
__global__ void Pad(cuDoubleComplex *input, cuDoubleComplex *output, int width, int width_out, int height) {
	
	int j = (blockIdx.x * blockDim.x + threadIdx.x)%width_out;
	int i = (blockIdx.x * blockDim.x + threadIdx.x)/width_out;
	
	output[blockIdx.x * blockDim.x + threadIdx.x].x = 0;
	output[blockIdx.x * blockDim.x + threadIdx.x].y = 0;
	
	//fill matrix with input data for even size
	if ((i<height)&&(j<width)) {
		output[(j+((width-1)/2))+(i+((height-1)/2))*width_out].x = input[j+i*width].x;
		output[(j+((width-1)/2))+(i+((height-1)/2))*width_out].y = input[j+i*width].y;
	}
}

/*-----------------------------------------------------------*/
/**
  * @brief Main entry point for this file.
  * @param  The matrix to convolute, the matrix width and height
  * and a pointer to the processing time.
  * @retval None
  */
extern "C" cuDoubleComplex *Conv_GPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time) {
	
	int width_out = width*2-1;
	int height_out = height*2-1;

    printf("\n%s Starting...", argv[0]);

    /* use command-line specified CUDA device, otherwise use device with highest Gflops/s */
    int devID = findCudaDevice(argc, (const char **)argv);
    
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    int size_padded = width_out*height_out;
    
    /* Set thread conditions */
    int num_threads = thread_limit;
    if (size_padded < thread_limit) {
		num_threads = size_padded;
    }
	
    unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded;
    
    /* setup execution parameters */
    /* calculate required grid dimensions */
    int grid_size = size_padded/thread_limit+1;
    dim3 grid(grid_size, 1, 1);
    dim3 threads(num_threads, 1, 1);      
    
    /* Move input data to device */
    cuDoubleComplex *data_in;
    checkCudaErrors(cudaMalloc((void **)&data_in, sizeof(cuDoubleComplex)*width*height));
    checkCudaErrors(cudaMemcpy(data_in, data, sizeof(cuDoubleComplex)*width*height, cudaMemcpyHostToDevice));
    
    /* Create zero padded matrix */
    cuDoubleComplex *data_padded;
    checkCudaErrors(cudaMalloc((void **)&data_padded, mem_size));
    Pad<<<grid,threads>>>(data_in,data_padded, width, width_out, height);
		
	/* Perform Convolution */
    cuDoubleComplex *h_convolved_signal;
    checkCudaErrors(cudaMalloc((void **)&h_convolved_signal, mem_size));
	Convolve<<<grid,threads>>>(data_padded, data_padded, size_padded, h_convolved_signal);
	
	/* Move result back to host */
	cuDoubleComplex *conv_out = (cuDoubleComplex*) malloc(mem_size);
	checkCudaErrors(cudaMemcpy(conv_out, h_convolved_signal, mem_size, cudaMemcpyDeviceToHost));
	                 
    sdkStopTimer(&timer);
    *time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    /* Free memory */
    cudaFree(data_in);
    cudaFree(data_padded);
    cudaFree(h_convolved_signal);        
    
    cudaDeviceReset();
    
    return(conv_out);
}



