#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

// 优化版本：使用 OpenMP 并行 + 循环展开 + 更好的缓存利用
template <typename T>
void linear_impl(T* out, const T* in, const T* weight, const T* bias,
                 size_t batch, size_t in_features, size_t out_features) {
    // 并行化外层循环（batch 和 out_features）
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            
            // 指针优化，减少地址计算
            const T* in_row = in + i * in_features;
            const T* weight_row = weight + j * in_features;
            
            // 循环展开 4 倍，提高指令级并行（ILP）
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                float x0 = llaisys::utils::cast<float>(in_row[k]);
                float x1 = llaisys::utils::cast<float>(in_row[k+1]);
                float x2 = llaisys::utils::cast<float>(in_row[k+2]);
                float x3 = llaisys::utils::cast<float>(in_row[k+3]);
                
                float w0 = llaisys::utils::cast<float>(weight_row[k]);
                float w1 = llaisys::utils::cast<float>(weight_row[k+1]);
                float w2 = llaisys::utils::cast<float>(weight_row[k+2]);
                float w3 = llaisys::utils::cast<float>(weight_row[k+3]);
                
                sum += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
            }
            
            // 处理剩余元素
            for (; k < in_features; ++k) {
                float x = llaisys::utils::cast<float>(in_row[k]);
                float w = llaisys::utils::cast<float>(weight_row[k]);
                sum += x * w;
            }
            
            if (bias) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * out_features + j] = llaisys::utils::cast<T>(sum);
        }
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
