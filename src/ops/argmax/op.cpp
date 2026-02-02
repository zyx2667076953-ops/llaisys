#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());
    // TODO: Support GPU
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
