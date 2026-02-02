#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t seqlen = q->shape()[0];
        size_t nhead = q->shape()[1];
        size_t d = q->shape()[2];
        
        size_t total_len = k->shape()[0];
        size_t nkvhead = k->shape()[1];
        
        size_t dv = v->shape()[2];
        
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
    }

    core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    // TODO: Support GPU
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
