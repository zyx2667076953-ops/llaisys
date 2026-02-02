from ctypes import (
    Structure,
    POINTER,
    c_int64,
    c_float,
    c_size_t,
    c_int,
)
from . import LIB_LLAISYS, llaisysTensor_t, llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


class LlaisysQwen2Model(Structure):
    pass


def load_qwen2(lib):
    """加载 Qwen2 相关函数签名"""
    # 函数声明
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int
    ]
    lib.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)

    lib.llaisysQwen2ModelDestroy.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [POINTER(LlaisysQwen2Model), POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelReset.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelReset.restype = None


# 在模块加载时初始化
load_qwen2(LIB_LLAISYS)
