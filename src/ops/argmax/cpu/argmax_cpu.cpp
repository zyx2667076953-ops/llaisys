#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <limits>

namespace llaisys::ops::cpu {

template <typename T>
void argmax_impl(int64_t* idx_out, T* val_out, const T* data, size_t n) {
    if (n == 0) return;
    
    T best = data[0];
    int64_t best_i = 0;
    
    for (size_t i = 1; i < n; ++i) {
        if (llaisys::utils::cast<float>(data[i]) > llaisys::utils::cast<float>(best)) {
            best = data[i];
            best_i = i;
        }
    }
    
    *idx_out = best_i;
    *val_out = best;
}

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, 
            llaisysDataType_t dtype, size_t numel) {
    int64_t* idx = reinterpret_cast<int64_t*>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_impl<float>(idx, reinterpret_cast<float*>(max_val), 
                           reinterpret_cast<const float*>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_impl<fp16_t>(idx, reinterpret_cast<fp16_t*>(max_val), 
                            reinterpret_cast<const fp16_t*>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_impl<bf16_t>(idx, reinterpret_cast<bf16_t*>(max_val), 
                             reinterpret_cast<const bf16_t*>(vals), numel);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
