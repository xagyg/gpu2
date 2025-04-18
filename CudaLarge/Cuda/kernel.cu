#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <memory.h>


const int numPeople = 5000;
const int loopCount = 10000000;  // Simulated heavy computation

__global__ void computePay(float* hours, float* rate, float* pay, int numPeople) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPeople) {
        float p = hours[idx] * rate[idx];
        for (int j = 0; j < loopCount; j++) {
            p *= 1.0000001f;
            p /= 1.0000001f;
        }
        pay[idx] = p;
    }
}

int main() {
    float* h_hoursWorked = (float*)malloc(numPeople * sizeof(float));
    float* h_hourlyRate = (float*)malloc(numPeople * sizeof(float));
    float* h_pay = (float*)malloc(numPeople * sizeof(float));

    // Fill with random values
    srand((unsigned int)time(NULL));
    for (int i = 0; i < numPeople; i++) {
        h_hoursWorked[i] = 16 + rand() % 32;               // 16–47 hours
        h_hourlyRate[i] = 15 + rand() % 16;               // $15–$30
    }

    clock_t start, end;
    double cpu_time_used;
    start = clock();  // Start timing

    // Allocate device memory
    float* d_hoursWorked, * d_hourlyRate, * d_pay;
    cudaMalloc(&d_hoursWorked, numPeople * sizeof(float));
    cudaMalloc(&d_hourlyRate, numPeople * sizeof(float));
    cudaMalloc(&d_pay, numPeople * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_hoursWorked, h_hoursWorked, numPeople * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hourlyRate, h_hourlyRate, numPeople * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPeople + threadsPerBlock - 1) / threadsPerBlock;

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);

    computePay << <blocksPerGrid, threadsPerBlock >> > (d_hoursWorked, d_hourlyRate, d_pay, numPeople);

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);

    //float milliseconds = 0;
   // cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("GPU time: %.3f seconds\n", milliseconds / 1000.0f);

    // Copy result back
    cudaMemcpy(h_pay, d_pay, numPeople * sizeof(float), cudaMemcpyDeviceToHost);

    end = clock();  // End timing
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the result
    printf("Pay for each person:\n");
    for (int i = 0; i < numPeople; i++) {
        printf("Person %d: $%.2f\n", i, h_pay[i]);
    }

    printf("Time taken: %f seconds\n", cpu_time_used);

    // Clean up
    cudaFree(d_hoursWorked);
    cudaFree(d_hourlyRate);
    cudaFree(d_pay);
    free(h_hoursWorked);
    free(h_hourlyRate);
    free(h_pay);

    return 0;
}
