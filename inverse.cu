#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

//can a function in cuda be called from the the cpp file :( 

using namespace std;

#define blocksize 64

extern "C"
double* calculateInverse(double *arr[], int len);

__global__ void nodiag_normalize(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
		if (x == i && x != y) {
			I[x*n + y] /= A[i*n + i];
			A[x*n + y] /= A[i*n + i];
		}

}

__global__ void diag_normalize(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
		if (x == y && x == i) {
			I[x*n + y] /= A[i*n + i];
			A[x*n + y] /= A[i*n + i];
		}

}

__global__ void gaussjordan(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		if (x != i) {
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i) {
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}
		}
	}

}

__global__ void set_zero(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		if (x != i) {
			if (y == i) {
				A[x*n + y] = 0;
			}
		}
	}
}

double* calculateInverse(double *L, int n)
{
	//flattening matrix
	double *iL = new double[n*n];
	
	double *d_A, *d_L, *I, *dI;
	
	float time;
	cudaError_t err;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int ddsize = n * n * sizeof(double);

	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	// memory allocation    
	err = cudaMalloc((void**)&d_A, ddsize);
	if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMalloc((void**)&dI, ddsize);
	if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	
	//making identity matrix
	I = new double[n*n];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<n; j++) {
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}

	//copy data from CPU to GPU
	err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

	//timer start
	cudaEventRecord(start, 0);

	// L^(-1)    
	for (int i = 0; i<n; i++) {
		nodiag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
		diag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
		gaussjordan <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
		set_zero <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy data from GPU to CPU
	err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
	err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

	cout << "Cuda Time - inverse: " << time << "ms\n";
	cudaFree(d_A);
	cudaFree(dI);

	double *c = new double[n*n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			c[i*n + j] = 0;  //put the initial value to zero
			for (int x = 0; x < n; x++)
				c[i*n + j] = c[i*n + j] + L[i*n + x] * iL[x*n + j];  //matrix multiplication
			cout << c[i*n + j] << endl;
		}
	}

	/*for (int i = 0; i < n; i++){
		double *row = new double[n];
		for (int j = 0 ; j < n; j++){
			row[j] = iL[i*n + j];	
		}
		inv[i] = *row;
	}*/
	delete[]I;
	delete[]L;
	delete[]iL;
	return c;
	
}