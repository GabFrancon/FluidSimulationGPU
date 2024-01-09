#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
  
std::size_t constexpr N = 10;

__global__ void my_kernel(float *const in)
{
    auto const i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
    {
        return;
    }

    in[i] = 1.0f;
}