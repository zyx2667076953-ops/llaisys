#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <limits>

namespace llaisys::ops::cpu {

template <typename T>
void argmax_impl(int64_t* max_idx, T* max_val, const T* vals, size_t numel) {
    if (numel == 0) return;

    T best_val = vals[0];
    int64_t best_idx = 0;

    for (size_t i = 1; i < numel; ++i) {
        float current = llaisys::utils::cast<float>(vals[i]);
        float best = llaisys::utils::cast<float>(best_val);
        if (current > best) {
            best_val = vals[i];
            best_idx = static_cast<int64_t>(i);
        }
    }

    *max_idx = best_idx;
    *max_val = best_val;
}

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, llaisysDataType_t dtype, size_t numel) {
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_impl<float>(idx_ptr, reinterpret_cast<float*>(max_val), reinterpret_cast<const float*>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_impl<fp16_t>(idx_ptr, reinterpret_cast<fp16_t*>(max_val), reinterpret_cast<const fp16_t*>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_impl<bf16_t>(idx_ptr, reinterpret_cast<bf16_t*>(max_val), reinterpret_cast<const bf16_t*>(vals), numel);
        break;
    default:
        // Handle other types if necessary
        break;
    }
}

} // namespace llaisys::ops::cpu
