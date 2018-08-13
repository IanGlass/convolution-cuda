/**************************************************************************//**
* @file Autocorrelation.cpp
* @brief Main module for performing autocorrelation. 
 * Equipped to perform multiple calls of convolution on the CPU, GPU and
 * a Fast Fourier Transform on the GPU. Making call to run_program will execute 
 * all three forms of autocorrelation, taking width, height and input data as arguments.
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
#include <cufft.h>

//includes, cuda
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>

/* Global Variables ----------------------------------------------------------*/
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

/*-----------------------------------------------------------*/
/**
  * @brief Function to execute FFT on the CPU and GPU
  * @param  The array width and height, data pointer, and the result
  * using convolution on the GPU and CPU and using FFT shift on the GPU.
  * @retval None
  */
void run_programs(int argc, char **argv, int Width, int Height, cuDoubleComplex *data, 
cuDoubleComplex *Result_convGPU, cuDoubleComplex *Result_convCPU, cuDoubleComplex *Result_FFTshift) {
	/* calculate required dimensions of output results */
	int width_out = Width*2-1;
	int height_out = Height*2-1;
	/* calculate size after padding (important for rest of code) */
	int size_padded = width_out*height_out;
	unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded;
	
	/* Temporary storage for time values */
	float *Time = (float*)malloc(sizeof(float));
	
	/* Floats for comparing output matrices */
	float *convCPU = (float *)malloc(sizeof(float)*width_out*height_out);
	float *convGPU = (float *)malloc(sizeof(float)*width_out*height_out);
	float *FFTshift = (float *)malloc(sizeof(float)*width_out*height_out);
	
	/* Convolution on CPU */
	Result_convCPU = Conv_CPU_fn(argc, argv, data, Width, Height, Time);
	//Load results into floats * for direct comparison
	for (int a = 0; a < height_out; a++) {
		for (int b = 0; b < width_out; b++) {
			convCPU[b+a*width_out] = (float)Result_convCPU[b+a*width_out].x;
		}
	}
	printf("\nProcessing time: %f (ms) CPU",*Time);
	
	/* Convolution on GPU */
	Result_convGPU = Conv_GPU_fn(argc, argv, data, Width, Height, Time);
	/* Load results into floats * for direct comparison */
	for (int c = 0; c < height_out; c++) {
		for (int d = 0; d < width_out; d++) {
			convGPU[d+c*width_out] = (float)Result_convGPU[d+c*width_out].x;
		}
	}
	/* Perform comparison */
	bool convGPUCorrect = sdkCompareL2fe((float *)convGPU, (float *)convCPU, size_padded, 1e-5f);
	printf("\nProcessing time: %f (ms) GPU",*Time);
    
    /* FFT with shift on GPU */
    Result_FFTshift = FFT_Shift_GPU_fn(argc, argv, data, Width, Height, Time);
    /* Load results into floats * for direct comparison */
    for (int e = 0; e < height_out; e++) {
		for (int f = 0; f < width_out; f++) {
			FFTshift[f+e*width_out] = (float)Result_FFTshift[f+e*width_out].x;
		}
	}
	/* Perform comparison */
	bool FFTshiftCorrect = sdkCompareL2fe((float *)FFTshift, (float *)convCPU, size_padded, 1e-5f);
    printf("\nProcessing time: %f (ms) FFT",*Time);
    
    /* Print comparison results */
    if (convGPUCorrect) {
		printf("\nGPU convolution match");
	}
	if (FFTshiftCorrect) {
		printf("\nFFT shift match");
	}
	
	free(convCPU);
	free(convGPU);
	free(FFTshift);
	free(Time);
}

/*-----------------------------------------------------------*/
/**
  * @brief Main entry point for the program
  * @param  The array width and height, data pointer, and the result
  * using convolution on the GPU and CPU and using FFT shift on the GPU.
  * @retval None
  */
int main(int argc, char **argv) {
	pArgc = &argc;
    pArgv = argv;
    
    int width = 15;
    int height = 15;
    
    cuDoubleComplex *matrix = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*width*height);
	
	/* Initialize test matrix */
	for (int i = 0; i < (width*height); i++) {
		matrix[i].x = 1;
		matrix[i].y = 0;
	}	
	
	/* Calculate size required for output matrix, based on padding */
	int width_out = width*2-1;
	int height_out = height*2-1;
	int size_padded = width_out*height_out;
	unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded;
	
	/* Allocate storage for output matrices */
	cuDoubleComplex *CPU_Conv = (cuDoubleComplex *)malloc(mem_size);
	cuDoubleComplex *GPU_Conv = (cuDoubleComplex *)malloc(mem_size);
	cuDoubleComplex *GPU_FFTshift = (cuDoubleComplex *)malloc(mem_size);
	
	/* Execute all three implementations */
	run_programs(argc, argv, width, height, matrix, CPU_Conv, GPU_Conv, GPU_FFTshift);
	
	/* Free the allocated memory for storage, results aren't used only the time to compute */
	free(matrix);
	free(CPU_Conv);
	free(GPU_Conv);
	free(GPU_FFTshift);
    
    printf("\n");
    exit(EXIT_SUCCESS);
}







