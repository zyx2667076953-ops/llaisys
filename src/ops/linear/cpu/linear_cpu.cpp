#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void linear_impl(T* out, const T* in, const T* weight, const T* bias,
                 size_t batch, size_t in_features, size_t out_features) {
    int64_t n = static_cast<int64_t>(batch * out_features);
    
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < n; ++idx) {
        size_t i = idx / out_features;
        size_t j = idx % out_features;
        
        const T* x = in + i * in_features;
        const T* w = weight + j * in_features;
        
        float sum = 0.0f;
        for (size_t k = 0; k < in_features; ++k) {
            sum += llaisys::utils::cast<float>(x[k]) * llaisys::utils::cast<float>(w[k]);
        }
        if (bias) sum += llaisys::utils::cast<float>(bias[j]);
        
        out[i * out_features + j] = llaisys::utils::cast<T>(sum);
    }
}

void linear(std::byte* out, const std::byte* in, const std::byte* weight, const std::byte* bias,
            llaisysDataType_t dtype, size_t batch, size_t in_features, size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_impl<float>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), 
                           reinterpret_cast<const float*>(weight), reinterpret_cast<const float*>(bias), 
                           batch, in_features, out_features);
        break;
    case LLAISYS_DTYPE_F16:
        linear_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), reinterpret_cast<const fp16_t*>(in), 
                            reinterpret_cast<const fp16_t*>(weight), reinterpret_cast<const fp16_t*>(bias), 
                            batch, in_features, out_features);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), reinterpret_cast<const bf16_t*>(in), 
                             reinterpret_cast<const bf16_t*>(weight), reinterpret_cast<const bf16_t*>(bias), 
                             batch, in_features, out_features);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
