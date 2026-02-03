#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

// 优化版本：OpenMP 并行 + 使用更快的 sigmoid 近似
template <typename T>
void swiglu_impl(T* out, const T* gate, const T* up, size_t numel) {
    // 并行化计算
    // 使用 int64_t 作为循环变量，因为 MSVC OpenMP 要求有符号整数类型
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(numel); ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        
        // SwiGLU: out = up * (gate * sigmoid(gate))
        // sigmoid(x) = 1 / (1 + exp(-x))
        float swish = g / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * swish);
    }
}

void swiglu(std::byte* out, const std::byte* gate, const std::byte* up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_impl<float>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(gate), 
                           reinterpret_cast<const float*>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), reinterpret_cast<const fp16_t*>(gate), 
                            reinterpret_cast<const fp16_t*>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), reinterpret_cast<const bf16_t*>(gate), 
                             reinterpret_cast<const bf16_t*>(up), numel);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
