#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t batch = in->numel() / in->shape().back();
        size_t in_features = in->shape().back();
        size_t out_features = weight->shape()[0];
        
        const std::byte* bias_data = bias ? bias->data() : nullptr;
        
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), batch, in_features, out_features);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    // TODO: Support GPU
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
