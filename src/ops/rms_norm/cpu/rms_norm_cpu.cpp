#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_impl(T* out, const T* in, const T* weight, size_t nrows, size_t dim, float eps) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nrows); ++i) {
        const T* row_in = in + i * dim;
        T* row_out = out + i * dim;
        
        // compute sum of squares
        float ss = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float x = llaisys::utils::cast<float>(row_in[j]);
            ss += x * x;
        }
        
        float rms = 1.0f / std::sqrt(ss / dim + eps);
        
        // normalize
        for (size_t j = 0; j < dim; ++j) {
            float x = llaisys::utils::cast<float>(row_in[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            row_out[j] = llaisys::utils::cast<T>(x * rms * w);
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
