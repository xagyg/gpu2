#include <stdio.h>
#include <ctime>

// Simulate heavy computation for each person's pay calculation
float calculatePersonPay(float hours, float rate) {
    float pay = hours * rate;

    // Simulate long calculation
    for (int i = 0; i < 10000000; i++) {
        pay = pay * 1.0000001f;  // Prevent compiler optimization
        pay = pay / 1.0000001f;
    }

    return pay;
}

int main() {
    const int numPeople = 100;
    
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

    clock_t start, end;
    double cpu_time_used;
    start = clock();  // Start timing

    // Calculate pay for each person sequentially
    for (int i = 0; i < numPeople; i++) {
        pay[i] = calculatePersonPay(hoursWorked[i], hourlyRate[i%10]);
     //   printf("Person %d: $%.2f\n", i, pay[i]);
    }

    end = clock();  // End timing

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Print results
    printf("Pay for each person:\n");
    for (int i = 0; i < numPeople; i++) {
        printf("Person %d: $%.2f\n", i, pay[i]);
    }

    printf("Time taken: %f seconds\n", cpu_time_used);

    return 0;
}