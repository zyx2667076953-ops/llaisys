#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t weight_dim = weight->shape().back();
        size_t num_indices = index->numel();
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), num_indices, weight_dim);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    // TODO: Support GPU
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
