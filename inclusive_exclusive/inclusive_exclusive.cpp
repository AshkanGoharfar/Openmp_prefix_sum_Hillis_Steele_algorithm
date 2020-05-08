// inclusive_exclusive.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
// **********************************************************************


#include "pch.h"
#include <iostream>

using namespace std;


#define _CRT_SECURE_NO_WARNINGS


#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>

void omp_check();
void fill_array(int *a, unsigned int n);
void prefix_sum(int *a, unsigned int n);
void print_array(int *a, unsigned int n);
void more_efficient_prefix_sum(int *a, size_t n);
void hills_steele_prefix(int* a, size_t n);

int main(int argc, char *argv[]) {
	// Check for correct compilation settings
	omp_check();
	// Input N
	unsigned int n = 0;
	printf("[-] Please enter N: ");
	scanf_s("%uld\n", &n);
	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);

	double start = omp_get_wtime();

	// Fill array with numbers 1..n
	fill_array(a, n);
	
	// Print array
	//print_array(a, n);
	
	// ***************** Hillis and Steele algorithm
	hills_steele_prefix(a, n);
	
	// ***************** most effiecient algorithm
	prefix_sum(a, n);
	
	// more efficient prefix sum algorithm
	more_efficient_prefix_sum(a, n);
	
	// Print array
	
	cout << "__________________________________________________________________________________________________________" << endl;
	cout << "__________________________________________________________________________________________________________" << endl;
	cout << "__________________________________________________________________________________________________________" << endl;
	cout << "__________________________________________________________________________________________________________" << endl;

	//print_array(a, n);
	// Free allocated memory
	free(a);
	double end = omp_get_wtime();

	double elapsed = end - start;

	cout << "\n\n" << endl;
	printf("Elapsed Time parallel %f\n", elapsed);
	return EXIT_SUCCESS;
}


void prefix_sum(int *a, unsigned int n) {
	float *suma;
	int NUM_THREADS = 4;

	omp_set_nested(1);
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
	{
		const int ithread = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
		//cout << "\nithread : " << endl;
#pragma omp single
		{
			suma = new float[nthreads + 1];
			suma[0] = 0;
		}
		float sum = 0;
#pragma omp for schedule(static, 1)
		for (int i = 0; i < n; i++) {
			sum += a[i];
			a[i] = sum;
		}
		suma[ithread + 1] = sum;
#pragma omp barrier
		float offset = 0;
		for (int i = 0; i < (ithread + 1); i++) {
			offset += suma[i];
		}
#pragma omp for schedule(static)
		for (int i = 0; i < n; i++) {
			a[i] += offset;
		}
	}
	delete[] suma;
}

void more_efficient_prefix_sum(int *a, size_t n) {
	int j;
	int threadNum = 4;
	int step = (int)log(n);
	//printf("%d", a[j]);
	omp_set_num_threads(8);

	for (int i = 1; i < n; i *= 2) {
#pragma omp parallel for 
		for (j = n; j >= i; j--) {
			a[j] += a[j - i];
			//printf("%d", a[j]);
		}
	}
}

void hills_steele_prefix(int* a, size_t n) {
	int* aHelp = (int*)malloc(n * sizeof aHelp);
	for (int i = 0; i < n; ++i) {
		aHelp[i] = i + 1;
	}
	for (int d = 1; d <= log2(n) + 1; d++) {
# pragma omp parallel for
		for (int k = 0; k < n; k++) {
			if (k >= pow(2, d - 1)) {
				(aHelp)[k] = (a)[(int)(k - pow(2, d - 1))] + (a)[k];
			}
			else {
				(aHelp)[k] = (a)[k];
			}
		}
		int* aTmp = a;
		a = aHelp;
		aHelp = aTmp;
	}
}

void print_array(int *a, unsigned int n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}

void fill_array(int *a, unsigned int n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}

void omp_check() {
	printf("------------ Info -------------\n");
#ifdef _DEBUG
	printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
	printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
	printf("[-] Platform: x64\n");
#elif _M_IX86 
	printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
	printf("[-] OpenMP is on.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
	printf("[!] OpenMP is off.\n");
	printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
	printf("[-] Maximum threads: %d\n", omp_get_max_threads());
	printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}
