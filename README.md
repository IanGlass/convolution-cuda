# FFT-CUDA

This project illustrates the advantages of using a GPU over a CPU for large complex computations.
The full report can be found in FFT Report.pdf

Autocorrelation.cpp is the main source file, executing convolution on the CPU through Conv_CPU.CPP

Autocorrelation.cpp also handles loading the warps for FFT_Shift_GPU and Conv_GPU_fn, which performs autocorrelation and convolution on the GPU using CUDA

Additionally, a matlab code file is also included, which performs correlation on the CPU
