import time
import random
import numpy  # For optimized matrix multiplication
import matplotlib.pyplot as plt  # For plotting performance comparison

# Function to generate a square matrix of random integers
def generate_square_matrix(n):
    "Generates an n x n matrix with random integer values between 0 and 10."
    return [[random.randint(0, 10) for _ in range(n)] for _ in range(n)]

# Part (a): Standard matrix multiplication using nested loops
def matrix_multiply(A, B):
    "Performs standard matrix multiplication on two square matrices, A and B."
    n = len(A)
    result = [[0] * n for i in range(n)]  # Initialize the result matrix with zeros

    # Triple nested loop for matrix multiplication
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]  # Multiply and sum up entries
    return result

# Part (b): Benchmark the standard matrix multiplication for different matrix sizes
def benchmark_standard_multiplication(A, B):
    "Benchmarks standard matrix multiplication for different matrix sizes."
    results = []
    for n in range(1, 1000, 100):  # Range of sizes from 1 to 999
        start_time = time.time()
        matrix_multiply(A, B)  # Multiply matrices A and B
        end_time = time.time()

        results.append((n, end_time - start_time))  # Store time taken for each size
        print(f"Matrix size {n}x{n} took {end_time - start_time} seconds")
    
    return results

# Part (c): Benchmark NumPy matrix multiplication and plot results
def benchmark_numpy_multiplication(A, B):
    "Benchmarks NumPy matrix multiplication for different matrix sizes."
    results = []
    for n in range(1, 1000, 100):
        np_A = numpy.array(A)  # Convert list to NumPy array
        np_B = numpy.array(B)

        start_time = time.time()
        numpy.dot(np_A, np_B)  # Use NumPy's optimized dot product
        end_time = time.time()

        results.append((n, end_time - start_time))  # Store time for each matrix size
        print(f"Matrix size {n}x{n} took {end_time - start_time} seconds")
    
    return results

def plot_benchmark_results(standard_results, numpy_results):
    "Plots benchmark results comparing standard vs. NumPy matrix multiplication."
    sizes = [size for size, _ in standard_results]  # Extract matrix sizes
    standard_times = [time for _, time in standard_results]  # Standard multiplication times
    numpy_times = [time for _, time in numpy_results]  # NumPy multiplication times

    plt.plot(sizes, standard_times, label='Standard Matrix Multiplication')
    plt.plot(sizes, numpy_times, label='NumPy Matrix Multiplication')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Matrix Multiplication Performance Comparison')
    plt.show()

# Part (e): Strassen's algorithm (non-recursive version) for 2x2 matrices
def strassen_non_recursive(A, B):
    "Performs matrix multiplication on 2x2 matrices using Strassen's algorithm."
    # Elements of matrix A
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    # Elements of matrix B
    e, f = B[0][0], B[0][1]
    g, h = B[1][0], B[1][1]

    # Calculate Strassen's seven products
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (f - h)
    M4 = d * (g - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)

    # Compute the entries of the result matrix
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6

    return [[c00, c01], [c10, c11]]

# Part (e): Strassen's algorithm (recursive version) for 2x2 matrices
def strassen_recursive(A, B):
    "Recursively performs Strassen's matrix multiplication on 2x2 matrices."
    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]

    # Elements of matrix A
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    # Elements of matrix B
    e, f = B[0][0], B[0][1]
    g, h = B[1][0], B[1][1]

    # Calculate Strassen's seven products using recursive calls
    M1 = strassen_recursive([[a + d]], [[e + h]])[0][0]
    M2 = strassen_recursive([[c + d]], [[e]])[0][0]
    M3 = strassen_recursive([[a]], [[f - h]])[0][0]
    M4 = strassen_recursive([[d]], [[g - e]])[0][0]
    M5 = strassen_recursive([[a + b]], [[h]])[0][0]
    M6 = strassen_recursive([[c - a]], [[e + f]])[0][0]
    M7 = strassen_recursive([[b - d]], [[g + h]])[0][0]
    
    # Compute the entries of the result matrix
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6

    return [[c00, c01], [c10, c11]]

# Benchmark function to evaluate performance of different matrix multiplication functions
def benchmark(function, A, B):
    "Benchmarks a given matrix multiplication function over multiple iterations."
    start_time = time.time()
    for _ in range(1000):  # Perform the multiplication 1000 times
        function(A, B)
    end_time = time.time()
    return end_time - start_time

# Main function to execute benchmarks and display results
def main():
    # Example matrix for benchmarking and testing (2x2 size for Strassen)
    A = [[1, 2], [1, 2]]
    B = [[1, 3], [0, 3]]

    # Testing Strassen's algorithm with non-recursive and recursive implementations
    print("Strassen Non-Recursive:", benchmark(strassen_non_recursive, A, B))
    print("Strassen Recursive:", benchmark(strassen_recursive, A, B))
    
    # Benchmark standard matrix multiplication and NumPy multiplication
    print("Standard Multiplication:", benchmark_standard_multiplication(A, B))
    print("NumPy Multiplication:", benchmark_numpy_multiplication(A, B))

if __name__ == "__main__":
    main()
