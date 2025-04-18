using System;
using System.Diagnostics;
using System.Threading.Tasks;

class Program
{
    static float CalculatePersonPay(float hours, float rate)
    {
        float pay = hours * rate;

        // Simulate long computation
        for (int i = 0; i < 10_000_000; i++)
        {
            pay *= 1.0000001f;
            pay /= 1.0000001f;
        }

        return pay;
    }

    static void Main()
    {
        const int numPeople = 100;

        float[] hoursWorked = new float[100]
        {
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

        float[] hourlyRate = new float[10] { 20, 25, 22, 18, 30, 15, 19, 28, 24, 21 };
        float[] pay = new float[numPeople];

        Stopwatch stopwatch = Stopwatch.StartNew();

        // Parallelized loop for faster computation
        Parallel.For(0, numPeople, i =>
        {
            pay[i] = CalculatePersonPay(hoursWorked[i % 100], hourlyRate[i % 10]);
        });

        stopwatch.Stop();

        // Print results
        Console.WriteLine("Pay for each person:");
        for (int i = 0; i < numPeople; i++)
        {
            Console.WriteLine($"Person {i}: ${pay[i]:F2}");
        }

        Console.WriteLine($"Time taken: {stopwatch.Elapsed.TotalSeconds:F4} seconds");
    }
}
