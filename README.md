# Convolution-CUDA
This project provides an overview of the processing performed on a GPU, CPU-GPU interaction and the advantage of using a GPU for certain processes. The example used is an FFT, however this overview will not provide a thorough explanation of how FFT works, rather focusing on GPU concepts.
The full report can be found in FFT-Report.pdf

## The GPU (Section 2.1)
Contrary to a CPU which has several high speed cores, a GPU consists of many modest speed cores to perform concurrent processing on shared memory. The *kernel* is the highest level software implementation on the GPU and acts as a CPU executable function. The *kernel* consists of a *grid* with pre-determined constituent thread *blocks*. A *block* is an array of *threads* executed in parallel, capable of inter-thread communication through a shared memory structure and is an instance of a single execution of the *kernel* code. In other words, the *kernel* defines an instruction set and each *thread* executes the instruction set on a different piece of memory, allowing concurrent processing.

<p align="center">
<img src="https://github.com/IanGlass/FFT-CUDA-Matlab/blob/master/Images/GPU-Structure.jpg" width="500">
</p>

The internal GPU memory structure consists of three main memory locations:
* Local thread memory - like cached memory in CPU for a single thread to store temporary data.
* Shared memory - Shared memory between thread within a single block for inter-thread communication.
* Global memory - Which acts like a buffer to allow the CPU to load data from RAM into the GPU for processing and allow processed memory to be passed back to the CPU.

<p align="center">
<img src="https://github.com/IanGlass/FFT-CUDA-Matlab/blob/master/Images/GPU-Memory.jpg" width="500">
</p>

More information can be found [here.](https://www.arc.vt.edu/resources/software/cuda/)

## Code

The main module provides the user with a function called ‘run_programs’, which takes an input matrix, dimensions and three pointers to store the results of an FFT on the GPU and convolution on the GPU and CPU. Complementary to the output results, the processing time for each method is displayed on the terminal as a unit of milliseconds. Users are not required to deal with more intricate details, such as block and grid size, as these are autonomously calculated. Autocorrelation can be defined as a cross-correlation of a signal with itself, using ‘self-convolution’. Truncating to one dimension, respective elements in a strip are multiplied and the result is accumulated to produce one element in the auto-correlated matrix. 

```c
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
	/* Load results into floats * for direct comparison */
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
```
### CPU Convolution
The first function called performs convolution of the input matrix on the CPU using the sliding strip method. A timer is used to measure the time taken to compute the result which is stored in the provided *time* pointer for *run_programs()* to use.
CPU conv

```c
/*-----------------------------------------------------------*/
/**
  * @brief Main function for this file.
  * @param  The data to be convoluted, the data width and height
  * and a pointer to the processing time.
  * @retval The convoluted data.
  */
extern "C" cuDoubleComplex *Conv_CPU_fn(int argc, char **argv, cuDoubleComplex *data, int width, int height, float *time) {

	StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
	
	int width_out = width*2-1;
	int height_out = height*2-1;
	int size_padded = width_out*height_out;
	unsigned int mem_size = sizeof(cuDoubleComplex)*size_padded; 
		
    /* create zero padded matrix before filling with input data */
    cuDoubleComplex *data_padded = (cuDoubleComplex *) malloc(mem_size);
    for (int z = 0; z < size_padded; z++) {
		data_padded[z].x = 0;
		data_padded[z].y = 0;
	}
	
	/* Fill matrix with input data and shift to center */
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j ++) {
			data_padded[(j+((width-1)/2))+(i+((height-1)/2))*width_out].x = data[j+i*width].x;
			data_padded[(j+((width-1)/2))+(i+((height-1)/2))*width_out].y = data[j+i*width].y;
		}
	}
	
	/* Allocate host memory for the convolution result */
    cuDoubleComplex *conv_out = (cuDoubleComplex *)malloc(mem_size);
	/* Convolve on the host */
    Convolve(data_padded, data_padded, size_padded, conv_out);
    
    sdkStopTimer(&timer);
    *time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    /* Free allocated memory */
    free(data_padded);
    
    return(conv_out);
}
```
### GPU Convolution
As the name suggests *Conv_GPU_fn()* performs convolution of the provided matrix on the GPU, implementing a timer to measure the processing time. This is then passed back into a *time* pointer for comparison to CPU processing speed.
```c
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
```

## Results (Section 2.3)

For large matrices (140x140) we see that the processing time required is ~6 s on the CPU compared with 0.13 s on the GPU, with the CPU run time increasing rapidly with size. 
<p align="center">
<img src="https://github.com/IanGlass/FFT-CUDA-Matlab/blob/master/Images/CPU-plot.jpg" width="500">
<img src="https://github.com/IanGlass/FFT-CUDA-Matlab/blob/master/Images/GPU-plot.jpg" width="500">
</p>





