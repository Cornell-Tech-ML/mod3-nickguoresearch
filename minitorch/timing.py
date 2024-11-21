from typing import Any
import minitorch
import time
import numpy as np
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: Any, size: int = 16) -> Any:
    """Run matmul"""
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y
    return z


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    sizes = [64, 128, 256, 512, 1024]
    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
    
    # Simplified Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, [times[size]["fast"] for size in sizes], 
             label='CPU Backend', color='blue')
    plt.plot(sizes, [times[size]["gpu"] for size in sizes], 
             label='GPU Backend', color='red')
    
    plt.title('Matrix Multiplication: CPU vs GPU Performance')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()