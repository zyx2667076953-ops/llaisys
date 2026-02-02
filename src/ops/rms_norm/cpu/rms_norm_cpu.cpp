#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

// 优化版本：OpenMP 并行 + 循环融合 + 更好的数值稳定性
template <typename T>
void rms_norm_impl(T* out, const T* in, const T* weight, size_t num_rows, size_t dim, float eps) {
    // 并行化行维度
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_rows; ++i) {
        const T* in_row = in + i * dim;
        T* out_row = out + i * dim;
        
        // 1. 计算平方和（循环展开）
        float sum_sq = 0.0f;
        size_t j = 0;
        
        // 展开 4 倍
        for (; j + 4 <= dim; j += 4) {
            float x0 = llaisys::utils::cast<float>(in_row[j]);
            float x1 = llaisys::utils::cast<float>(in_row[j + 1]);
            float x2 = llaisys::utils::cast<float>(in_row[j + 2]);
            float x3 = llaisys::utils::cast<float>(in_row[j + 3]);
            sum_sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        }
        
        // 处理剩余
        for (; j < dim; ++j) {
            float x = llaisys::utils::cast<float>(in_row[j]);
            sum_sq += x * x;
        }
        
        // 2. 计算 RMS 的倒数（避免除法）
        float inv_rms = 1.0f / std::sqrt(sum_sq / dim + eps);
        
        // 3. 归一化并应用权重（循环融合）
        for (j = 0; j < dim; ++j) {
            float x = llaisys::utils::cast<float>(in_row[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            out_row[j] = llaisys::utils::cast<T>(w * x * inv_rms);
        }
    }
}

void rms_norm(std::byte* out, const std::byte* in, const std::byte* weight,
              llaisysDataType_t dtype, size_t num_rows, size_t dim, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_impl<float>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), 
                             reinterpret_cast<const float*>(weight), num_rows, dim, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), reinterpret_cast<const fp16_t*>(in), 
                              reinterpret_cast<const fp16_t*>(weight), num_rows, dim, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), reinterpret_cast<const bf16_t*>(in), 
                               reinterpret_cast<const bf16_t*>(weight), num_rows, dim, eps);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
