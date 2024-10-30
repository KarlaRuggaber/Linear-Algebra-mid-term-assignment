import time
import random
import numpy 
import matplotlib.pyplot as plt 

#part (a) matrix multiplication
def matrix_multiply(A, B):
    n = len(A)
    result = [[0]*n for i in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

#part (b)
def benchmark_standard_multiplication(A, B):
    results = []
    for n in range(1, 1000, 100):
        start_time = time.time()
        matrix_multiply(A, B)
        end_time = time.time()

        results.append((n, end_time - start_time))
        print(f"Matrix size {n}x{n} took {end_time - start_time} seconds")
    
    return results

#part (c): comparison with NumPy and Plotting
def benchmark_numpy_multiplication(A, B):
    results = []
    for n in range(1, 1000, 100):
        np_A = numpy.array(A)
        np_B = numpy.array(B)

        start_time = time.time()
        numpy.dot(np_A, np_B)
        end_time = time.time()

        results.append((n, end_time - start_time))
        print(f"Matrix size {n}x{n} took {end_time - start_time} seconds")
    
    return results

def plot_benchmark_results(standard_results, numpy_results):
    sizes = [size for size, i in standard_results]
    standard_times = [times for j, times in standard_results]
    numpy_times = [times for k, times in numpy_results]

    plt.plot(sizes, standard_times, label='Standard Matrix Multiplication')
    plt.plot(sizes, numpy_times, label='NumPy Matrix Multiplication')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Matrix Multiplication Performance Comparison')
    plt.show()

def strassen_non_recursive(A, B):
    #elements of A
    a = A[0][0]
    b = A[0][1]
    c = A[1][0]
    d = A[1][1]

    #elements of B
    e = B[0][0]
    f = B[0][1]
    g = B[1][0]
    h = B[1][1]

    M1 = (a + d)*(e + h)
    M2 = (c + d)*e
    M3 = a*(f - h)
    M4 = d*(g - e)
    M5 = (a + b)*h
    M6 = (c - a)*(e + f)
    M7 = (b - d)*(g + h)

    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6

    return [[c00, c01], [c10, c11]]

def strassen_recursive(A, B):
    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]
    
    #elements of A
    a = A[0][0]
    b = A[0][1]
    c = A[1][0]
    d = A[1][1]

    #elements of B
    e = B[0][0]
    f = B[0][1]
    g = B[1][0]
    h = B[1][1]

    M1 = strassen_recursive([[a + d]], [[e + h]])[0][0]
    M2 = strassen_recursive([[c + d]], [[e]])[0][0]
    M3 = strassen_recursive([[a]], [[f - h]])[0][0]
    M4 = strassen_recursive([[d]], [[g - e]])[0][0]
    M5 = strassen_recursive([[a + b]], [[h]])[0][0]
    M6 = strassen_recursive([[c - a]], [[e + f]])[0][0]
    M7 = strassen_recursive([[b - d]], [[g + h]])[0][0]
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6

    return [[c00, c01], [c10, c11]]

def benchmark(functions, A, B):
    start_time = time.time()
    for i in range(1000):
        functions(A, B)
    end_time = time.time()
    return end_time - start_time


def main():
    #plot_benchmark_results(benchmark_standard_multiplication(),benchmark_numpy_multiplication())
    A = [[1, 2], [1, 2]]
    B = [[1, 3], [0, 3]]
    #print(strassen_non_recursive(A, B))
    #print(strassen_recursive(A, B))

    print(benchmark(strassen_non_recursive, A, B))
    print(benchmark(strassen_recursive, A, B))
    print(benchmark_standard_multiplication(A, B))
    print(benchmark_numpy_multiplication(A, B))

if __name__ == "__main__":
    main()



