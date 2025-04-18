#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <memory.h>

#include <stdio.h>

__device__ float calculatePersonPay(float hours, float rate) {
    float pay = hours * rate;

    // Fake heavy computation
    for (int i = 0; i < 10000000; i++) {
        pay = pay * 1.0000001f;  // Small operation to avoid compiler optimization
        pay = pay / 1.0000001f;
    }

    return pay;
}

// CUDA kernel to calculate pay
__global__ void calculatePay(float* hoursWorked, float* hourlyRate, float* pay) {
    int idx = threadIdx.x; // Thread ID corresponds to person ID

    pay[idx] = calculatePersonPay(hoursWorked[idx%100], hourlyRate[idx%10]);
}


int main() {
    
    const int numPeople = 100;

    // Host (CPU) arrays
    float hoursWorked[numPeople] = {
        32, 36, 22, 28, 24, 38, 40, 35, 29, 33,
        41, 27, 39, 20, 26, 37, 30, 42, 34, 25,
        18, 44, 31, 23, 38, 19, 21, 40, 36, 28,
        22, 45, 26, 33, 39, 21, 35, 43, 20, 30,
        47, 37, 24, 32, 29, 31, 25, 22, 36, 38,
        19, 34, 40, 33, 30, 21, 43, 27, 35, 26,
        41, 28, 18, 37, 24, 20, 22, 29, 34, 31,
        23, 36, 44, 39, 32, 25, 28, 19, 40, 46,
        27, 20, 35, 38, 33, 42, 30, 21, 45, 26,
        23, 34, 31, 17, 43, 29, 16, 39, 36, 30
    };
    float hourlyRate[10] = { 20, 25, 22, 18, 30, 15, 19, 28, 24, 21 };
    float pay[numPeople];

    // Device (GPU) arrays
    float* d_hoursWorked, * d_hourlyRate, * d_pay;

    clock_t start, end;
    double cpu_time_used;
    start = clock();  // Start timing

    // Allocate memory on GPU
    cudaMalloc((void**)&d_hoursWorked, numPeople * sizeof(float));
    cudaMalloc((void**)&d_hourlyRate, numPeople * sizeof(float));
    cudaMalloc((void**)&d_pay, numPeople * sizeof(float));

    // Copy data from Host to Device
    cudaMemcpy(d_hoursWorked, hoursWorked, numPeople * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hourlyRate, hourlyRate, numPeople * sizeof(float), cudaMemcpyHostToDevice);

    // start = clock();  // Start timing (when excluding memory transfers)

    // Launch Kernel with 1 block of 100 threads
    calculatePay <<<1, numPeople >> > (d_hoursWorked, d_hourlyRate, d_pay);

    // Copy result back to Host
    cudaMemcpy(pay, d_pay, numPeople * sizeof(float), cudaMemcpyDeviceToHost);

    end = clock();  // End timing
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the result
    printf("Pay for each person:\n");
    for (int i = 0; i < numPeople; i++) {
        printf("Person %d: $%.2f\n", i, pay[i]);
    }

    printf("Time taken: %f seconds\n", cpu_time_used);

    // Free GPU memory
    cudaFree(d_hoursWorked);
    cudaFree(d_hourlyRate);
    cudaFree(d_pay);

    return 0;
}
