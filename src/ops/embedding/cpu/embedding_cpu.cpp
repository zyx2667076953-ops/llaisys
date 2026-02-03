#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void embedding_impl(T* out, const int64_t* idx, const T* weight, size_t n, size_t dim) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        int64_t row = idx[i];
        std::memcpy(out + i * dim, weight + row * dim, dim * sizeof(T));
    }
}

void embedding(std::byte* out, const std::byte* index, const std::byte* weight, 
               llaisysDataType_t dtype, size_t num_indices, size_t weight_dim) {
    const int64_t* idx = reinterpret_cast<const int64_t*>(index);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_impl<float>(reinterpret_cast<float*>(out), idx, 
                              reinterpret_cast<const float*>(weight), num_indices, weight_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_impl<fp16_t>(reinterpret_cast<fp16_t*>(out), idx, 
                               reinterpret_cast<const fp16_t*>(weight), num_indices, weight_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_impl<bf16_t>(reinterpret_cast<bf16_t*>(out), idx, 
                                reinterpret_cast<const bf16_t*>(weight), num_indices, weight_dim);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::cpu
