#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

//bool test_equality(double* arr1 [], double* arr2[]){
//	int n = sizeof(arr1)/sizeof(*arr1);
//	if (n == sizeof(arr2)/sizeof(*arr2)){
//		for (int i = 0; i < n; i++){
//			for (int j = 0; j < n; j++){
//				if (arr1[i][j] != arr2[i][j]){
//					return false;
//				}
//			}
//		}
//	}
//}

extern "C"
double* calculateInverse(double *arr[], int len);

bool test_norm(double *arr1, double arr2[], int n) {
	double norm = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			arr1[0];
			double diff = (&arr1)[i][j] - (&arr2)[i][j];
			norm += diff * diff;
		}
	}
	return sqrt(norm) < 1e-6;
}

int main() {

	double **arr1;
	arr1 = new double*[3];
	double *arr1a = new double[3];
	arr1a[0] = 0.5;
	arr1a[1] = 3;
	arr1a[2] = 4;
	arr1[0] = arr1a;
	double *arr1b= new double[3];
	arr1b[0] = 1;
	arr1b[1] = 3;
	arr1b[2] = 10;
	arr1[1] = arr1b;
	double *arr1c = new double[3];
	arr1c[0] = 4;
	arr1c[1] = 9;
	arr1c[2] = 16;
	arr1[2] = arr1c;
	//	// each i-th pointer is now pointing to dynamic array (size 10)
	////	// of actual int values
	////
	////double arr1[3][3] = {{0.5, 3, 4},
	////				{1, 3, 10},
	////				{4 , 9, 16 }};
	////				
	////double arr2[3][3] = {{0, 3, 4},
	////				{1, 3, 10},
	////				{4 , 9, 16 }};
	////				
	////double arr3[3][3] = {{0, 3, 4},
	////				{1, 5, 6},
	////				{9 , 8, 2}};
	////				
	////double arr4[3][3] = {{22, 3, 4},
	////				{1, 5, 6},
	////				{9 , 8, 2 }};
	//				

	double *arr1res = calculateInverse(arr1, 3);


	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			std::cout << *arr1res++ << std::endl;;
		}
	}
	///*double *arr2res = calculateInverse(arr2, 3);
	//double *arr3res = calculateInverse(arr3, 3);
	//double *arr4res = calculateInverse(arr4, 3);*/
	//
	////
	//double arr1inv[3][3] = {{-0.700000, -0.200000, 0.300000},
	//				{0.400000, -0.266667, -0.066667},
	//				{-0.050000 , 0.200000, -0.050000 }};
					
	//double arr2[3][3] = {{-1.076923, -0.307692, 0.461538},
	//				{-1.076923, -0.205128, -0.025641},
	//				{-0.076923 , 0.192308, -0.038462 }};
	//				
	//double arr3[3][3] = {
	//	{-4.750000     ,  3.250000    ,    -0.250000},
	//	{6.500000   ,     -4.500000 ,      0.500000},
	//{-4.625000 ,      3.375000,        -0.375000}};
	//				
	//double arr4[3][3] ={ {0.045894   ,     -0.031401     ,  0.002415},
	//	{-0.062802     ,  -0.009662 ,      0.154589},
	//		{0.044686       , 0.179952   ,     -0.129227}};
	//	
	//
	//std::cout << (test_norm(*arr1inv, arr1res,3));
	//std::cout << (test_equality(arr2inv, arr2res));
	//std::cout <<(test_equality(arr3inv, arr3res));
	//std::cout<<(test_equality(arr4inv, arr4res));

	//might have to do linalgnorm or something in case of floating point errors
	return 0;
}