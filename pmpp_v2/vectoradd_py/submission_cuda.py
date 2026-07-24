#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu B200

from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = """
#include <cuda_fp16.h>

__global__ void vector_add_kernel(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n8 = n / 8;

    // 한 스레드가 8개의 half를 처리 (float4 = 4 * 4 = 16 = 8 * 2(__half size))
    if (idx < n8) {
        const float4 av = reinterpret_cast<const float4*>(a)[idx];
        const float4 bv = reinterpret_cast<const float4*>(b)[idx];
        const __half2* ah = reinterpret_cast<const __half2*>(&av);
        const __half2* bh = reinterpret_cast<const __half2*>(&bv);
        float4 cv;
        __half2* ch = reinterpret_cast<__half2*>(&cv);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ch[i] = __hadd2(ah[i], bh[i]);
        }
        reinterpret_cast<float4*>(c)[idx] = cv;
    }

    // 나머지 처리
    int tail = n8 * 8 + idx;
    if (tail < n) {
        const __half ai = a[tail];
        const __half bi = b[tail];
        c[tail] = __hadd(ai, bi);
    }
}

torch::Tensor add_vector(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    int n = x.numel();
    int threads = 256;
    int items = (n + 7) / 8;  // 스레드당 half 8개
    int blocks = (items + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(y.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return out;
}
"""

cpp_src = """
torch::Tensor add_vector(torch::Tensor x, torch::Tensor y, torch::Tensor out);
"""

mod = load_inline(
    name="add_vector_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_vector"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    return mod.add_vector(A, B, output)

# A100
# threads = 256, naive, 1203.712μs
# threads = 256, vectorized, 894.293μs

# B200
# threads = 256, 234.990μs
# threads = 128, 235.478μs
