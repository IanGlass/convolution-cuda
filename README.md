# FFT-CUDA
This project provides an overview of the processing performed on a GPU, CPU-GPU interaction and the advantage of using a GPU for certain processes. The example used is an FFT, however this overview will not provide a thorough explanation of how FFT works, rather focusing on GPU concepts.

<<<<<<< HEAD
## The GPU
Contrary to a CPU which has several high speed cores, a GPU consists of many modest speed cores to perform concurrent processing on shared memory. The *kernel* is the highest level software implementation on the GPU and acts as a CPU executable function. The *kernel* consists of a *grid* with pre-determined constituent thread *blocks*. A *block* is an array of *threads* executed in parallel, capable of inter-thread communication through a shared memory structure and is an instance of a single execution of the *kernel* code. In other words, the *kernel* defines an instruction set and each *thread* executes the instruction set on a different piece of memory, allowing concurrent processing.

<p align="center">
<img src="https://github.com/IanGlass/FFT-CUDA-Matlab/blob/master/GPU-Structure.jpg" width="500">
</p>

The internal GPU memory structure consists of three main memory locations:
* Local thread memory - like cached memory in CPU for a single thread to store temporary data.
* Shared memory - Shared memory between thread within a single block for inter-thread communication.
Global memory - Which acts like a buffer to allow the CPU to load data from RAM into the GPU for processing and allow processed memory to be passed back to the CPU.

<p align="center">
<img src="https://github.com/IanGlass/FFT-CUDA-Matlab/blob/master/GPU-Memory.jpg" width="500">
</p>

More information can be found [here](https://www.arc.vt.edu/resources/software/cuda/)

## Code

=======
>>>>>>> 762ddf14fe06cb136e37e78a95c9cf417ad6481f
This project illustrates the advantages of using a GPU over a CPU for large complex computations.
The full report can be found in FFT Report.pdf

Autocorrelation.cpp is the main source file, executing convolution on the CPU through Conv_CPU.CPP

Autocorrelation.cpp also handles loading the warps for FFT_Shift_GPU and Conv_GPU_fn, which performs autocorrelation and convolution on the GPU using CUDA

Additionally, a matlab code file is also included, which performs correlation on the CPU
