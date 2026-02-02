#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rope_impl(T* out, const T* in, const int64_t* pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half_d = d / 2;
    for (size_t i = 0; i < seqlen; ++i) {
        int64_t p_i = pos_ids[i];
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < half_d; ++j) {
                float phi = p_i / std::pow(theta, 2.0f * j / d);
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                size_t base_idx = i * nhead * d + h * d;
                float a = llaisys::utils::cast<float>(in[base_idx + j]);
                float b = llaisys::utils::cast<float>(in[base_idx + j + half_d]);
                
                out[base_idx + j] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out[base_idx + j + half_d] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

void rope(std::byte* out, const std::byte* in, const std::byte* pos_ids,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t d, float theta) {
    const int64_t* p_ids = reinterpret_cast<const int64_t*>(pos_ids);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_impl<float>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), p_ids, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), reinterpret_cast<const fp16_t*>(in), p_ids, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), reinterpret_cast<const bf16_t*>(in), p_ids, seqlen, nhead, d, theta);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
