#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

// 优化版本：OpenMP 并行（每个索引的查找是独立的）
template <typename T>
void embedding_impl(T* out, const int64_t* index, const T* weight, size_t num_indices, size_t weight_dim) {
    // 并行化索引查找
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_indices; ++i) {
        int64_t idx = index[i];
        std::memcpy(out + i * weight_dim, weight + idx * weight_dim, weight_dim * sizeof(T));
    }
}

void embedding(std::byte* out, const std::byte* index, const std::byte* weight, llaisysDataType_t dtype, size_t num_indices, size_t weight_dim) {
    const int64_t* idx_ptr = reinterpret_cast<const int64_t*>(index);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_impl<float>(reinterpret_cast<float*>(out), idx_ptr, reinterpret_cast<const float*>(weight), num_indices, weight_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), idx_ptr, reinterpret_cast<const fp16_t*>(weight), num_indices, weight_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), idx_ptr, reinterpret_cast<const bf16_t*>(weight), num_indices, weight_dim);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
