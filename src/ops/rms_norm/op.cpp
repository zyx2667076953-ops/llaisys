#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t dim = in->shape().back();
        size_t num_rows = in->numel() / dim;
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), num_rows, dim, eps);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    // TODO: Support GPU
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
