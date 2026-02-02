#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t seqlen = in->shape()[0];
        size_t nhead = in->shape()[1];
        size_t d = in->shape()[2];
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    // TODO: Support GPU
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
