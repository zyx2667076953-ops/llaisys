#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rope_impl(T* out, const T* in, const int64_t* pos, size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half = d / 2;
    
    for (size_t i = 0; i < seqlen; ++i) {
        int64_t p = pos[i];
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < half; ++j) {
                float freq = p / std::pow(theta, 2.0f * j / d);
                float c = std::cos(freq);
                float s = std::sin(freq);
                
                size_t base = i * nhead * d + h * d;
                float a = llaisys::utils::cast<float>(in[base + j]);
                float b = llaisys::utils::cast<float>(in[base + j + half]);
                
                out[base + j] = llaisys::utils::cast<T>(a * c - b * s);
                out[base + j + half] = llaisys::utils::cast<T>(b * c + a * s);
            }
        }
    }
}

void rope(std::byte* out, const std::byte* in, const std::byte* pos_ids,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t d, float theta) {
    const int64_t* pos = reinterpret_cast<const int64_t*>(pos_ids);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_impl<float>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), 
                         pos, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), reinterpret_cast<const fp16_t*>(in), 
                          pos, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), reinterpret_cast<const bf16_t*>(in), 
                          pos, seqlen, nhead, d, theta);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
