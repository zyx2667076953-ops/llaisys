#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops::cpu {

// 优化版本：OpenMP 并行 + 减少内存分配
template <typename T>
void self_attention_impl(T* attn_val, const T* q, const T* k, const T* v,
                        size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    size_t head_group_size = nhead / nkvhead;

    // 并行化 head 和 seqlen 两个维度
    // 使用 int64_t 作为循环变量，因为 MSVC OpenMP 要求有符号整数类型
    // MSVC OpenMP 2.0 不完全支持 collapse，改为并行化合并后的迭代空间
    int64_t total_iterations = static_cast<int64_t>(nhead * seqlen);
    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < total_iterations; ++idx) {
        int64_t h = idx / static_cast<int64_t>(seqlen);
        int64_t i = idx % static_cast<int64_t>(seqlen);
        size_t kv_h = h / head_group_size;
            
            // 使用栈分配替代 vector，减少动态内存分配开销
            float* scores = new float[total_len];
            float max_score = -std::numeric_limits<float>::infinity();

            // 1. Compute attention scores (Q·K^T)
            size_t valid_len = total_len - seqlen + i + 1; // causal mask 有效长度
            
            // 预先计算指针位置
            const T* q_ptr = q + i * nhead * d + h * d;
            
            for (size_t t = 0; t < valid_len; ++t) {
                float sum = 0.0f;
                const T* k_ptr = k + t * nkvhead * d + kv_h * d;
                
                // 循环展开 (unroll by 4)
                size_t k_idx = 0;
                for (; k_idx + 4 <= d; k_idx += 4) {
                    float q0 = llaisys::utils::cast<float>(q_ptr[k_idx]);
                    float q1 = llaisys::utils::cast<float>(q_ptr[k_idx + 1]);
                    float q2 = llaisys::utils::cast<float>(q_ptr[k_idx + 2]);
                    float q3 = llaisys::utils::cast<float>(q_ptr[k_idx + 3]);
                    
                    float k0 = llaisys::utils::cast<float>(k_ptr[k_idx]);
                    float k1 = llaisys::utils::cast<float>(k_ptr[k_idx + 1]);
                    float k2 = llaisys::utils::cast<float>(k_ptr[k_idx + 2]);
                    float k3 = llaisys::utils::cast<float>(k_ptr[k_idx + 3]);
                    
                    sum += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                }
                
                // 处理剩余元素
                for (; k_idx < d; ++k_idx) {
                    float q_val = llaisys::utils::cast<float>(q_ptr[k_idx]);
                    float k_val = llaisys::utils::cast<float>(k_ptr[k_idx]);
                    sum += q_val * k_val;
                }
                
                scores[t] = sum * scale;
                max_score = std::max(max_score, scores[t]);
            }
            
            // Causal masking：后面的位置设为 0
            for (size_t t = valid_len; t < total_len; ++t) {
                scores[t] = 0.0f;
            }

            // 2. Softmax (numerically stable)
            float sum_exp = 0.0f;
            for (size_t t = 0; t < valid_len; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            
            float inv_sum = 1.0f / sum_exp;
            for (size_t t = 0; t < valid_len; ++t) {
                scores[t] *= inv_sum;
            }

            // 3. Weighted sum with V
            T* out_ptr = attn_val + i * nhead * dv + h * dv;
            
            for (size_t v_idx = 0; v_idx < dv; ++v_idx) {
                float sum = 0.0f;
                for (size_t t = 0; t < valid_len; ++t) {
                    float v_val = llaisys::utils::cast<float>(v[t * nkvhead * dv + kv_h * dv + v_idx]);
                    sum += scores[t] * v_val;
                }
                out_ptr[v_idx] = llaisys::utils::cast<T>(sum);
            }
            
            delete[] scores;
    }
}

void self_attention(std::byte* attn_val, const std::byte* q, const std::byte* k, const std::byte* v,
                   llaisysDataType_t dtype, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl<float>(reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q), 
                                  reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v), 
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl<fp16_t>(reinterpret_cast<fp16_t*>(attn_val), reinterpret_cast<const fp16_t*>(q), 
                                   reinterpret_cast<const fp16_t*>(k), reinterpret_cast<const fp16_t*>(v), 
                                   seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl<bf16_t>(reinterpret_cast<bf16_t*>(attn_val), reinterpret_cast<const bf16_t*>(q), 
                                    reinterpret_cast<const bf16_t*>(k), reinterpret_cast<const bf16_t*>(v), 
                                    seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
