#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>
#include <omp.h>

namespace llaisys::ops::cpu {

// 优化版本：使用 OpenMP 并行 + 更好的缓存利用
template <typename T>
void linear_impl_optimized(T* out, const T* in, const T* weight, const T* bias,
                          size_t batch, size_t in_features, size_t out_features) {
    // 并行化外层循环（batch 和 out_features）
    // 使用 int64_t 作为循环变量，因为 MSVC OpenMP 要求有符号整数类型
    // MSVC OpenMP 2.0 不完全支持 collapse，改为并行化合并后的迭代空间
    int64_t total_iterations = static_cast<int64_t>(batch * out_features);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total_iterations; ++idx) {
        int64_t i = idx / static_cast<int64_t>(out_features);
        int64_t j = idx % static_cast<int64_t>(out_features);
            float sum = 0.0f;
            
            // 内积计算（最内层循环）
            const T* in_row = in + i * in_features;
            const T* weight_row = weight + j * in_features;
            
            // 循环展开 + 累加（提高 ILP）
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

} // namespace llaisys::ops::cpu
