#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_impl(T* out, const T* q, const T* k, const T* v,
                        size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, 
                        size_t d, size_t dv, float scale) {
    size_t gsize = nhead / nkvhead;
    int64_t n = static_cast<int64_t>(nhead * seqlen);
    
    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < n; ++idx) {
        size_t h = idx / seqlen;
        size_t i = idx % seqlen;
        size_t kv_h = h / gsize;
        
        std::vector<float> scores(total_len);
        float max_s = -1e9f;
        
        // causal mask
        size_t valid = total_len - seqlen + i + 1;
        
        const T* qp = q + i * nhead * d + h * d;
        
        // compute QK^T
        for (size_t t = 0; t < valid; ++t) {
            const T* kp = k + t * nkvhead * d + kv_h * d;
            float s = 0.0f;
            for (size_t x = 0; x < d; ++x) {
                s += llaisys::utils::cast<float>(qp[x]) * llaisys::utils::cast<float>(kp[x]);
            }
            scores[t] = s * scale;
            max_s = std::max(max_s, scores[t]);
        }
        
        // softmax
        float sum = 0.0f;
        for (size_t t = 0; t < valid; ++t) {
            scores[t] = std::exp(scores[t] - max_s);
            sum += scores[t];
        }
        for (size_t t = 0; t < valid; ++t) {
            scores[t] /= sum;
        }
        
        // weighted sum
        T* op = out + i * nhead * dv + h * dv;
        for (size_t x = 0; x < dv; ++x) {
            float acc = 0.0f;
            for (size_t t = 0; t < valid; ++t) {
                acc += scores[t] * llaisys::utils::cast<float>(v[t * nkvhead * dv + kv_h * dv + x]);
            }
            op[x] = llaisys::utils::cast<T>(acc);
        }
    }
}

void self_attention(std::byte* attn_val, const std::byte* q, const std::byte* k, const std::byte* v,
                   llaisysDataType_t dtype, size_t seqlen, size_t total_len, size_t nhead, 
                   size_t nkvhead, size_t d, size_t dv, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl<float>(reinterpret_cast<float*>(attn_val), 
                                  reinterpret_cast<const float*>(q), 
                                  reinterpret_cast<const float*>(k), 
                                  reinterpret_cast<const float*>(v), 
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl<fp16_t>(reinterpret_cast<fp16_t*>(attn_val), 
                                   reinterpret_cast<const fp16_t*>(q), 
                                   reinterpret_cast<const fp16_t*>(k), 
                                   reinterpret_cast<const fp16_t*>(v), 
                                   seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl<bf16_t>(reinterpret_cast<bf16_t*>(attn_val), 
                                    reinterpret_cast<const bf16_t*>(q), 
                                    reinterpret_cast<const bf16_t*>(k), 
                                    reinterpret_cast<const bf16_t*>(v), 
                                    seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
