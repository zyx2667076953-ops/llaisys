from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.qwen2 import (
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    LlaisysQwen2Model,
)
from ..tensor import Tensor

from pathlib import Path
import safetensors
import json
from ctypes import c_int, c_int64, POINTER, cast


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. 加载配置
        with open(model_path / "config.json", "r") as f:
            hf_config = json.load(f)
        
        # 2. 创建元数据
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16
        meta.nlayer = hf_config["num_hidden_layers"]
        meta.hs = hf_config["hidden_size"]
        meta.nh = hf_config["num_attention_heads"]
        meta.nkvh = hf_config.get("num_key_value_heads", meta.nh)
        meta.dh = meta.hs // meta.nh
        meta.di = hf_config["intermediate_size"]
        # 限制最大序列长度，避免 KV-Cache 内存过大
        # 原始 max_position_embeddings 可能是 131072，这会导致内存爆炸
        raw_maxseq = hf_config.get("max_position_embeddings", 2048)
        meta.maxseq = min(raw_maxseq, 4096)  # 限制为 4096
        meta.voc = hf_config["vocab_size"]
        meta.epsilon = hf_config.get("rms_norm_eps", 1e-6)
        meta.theta = hf_config.get("rope_theta", 10000.0)
        meta.end_token = hf_config.get("eos_token_id", 151643)
        
        print(f"Model config: nlayer={meta.nlayer}, hs={meta.hs}, nh={meta.nh}, nkvh={meta.nkvh}, dh={meta.dh}, di={meta.di}, voc={meta.voc}")
        
        # 3. 创建模型
        from ctypes import byref
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),  # 传递指针
            DeviceType(device),
            cast(None, POINTER(c_int)),  # device_ids (暂不支持多设备)
            0  # ndevice
        )
        
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")
        
        # 4. 获取权重结构
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not self._weights_ptr:
            raise RuntimeError("Failed to get model weights")
        
        self._weights = self._weights_ptr.contents
        self._meta = meta
        
        # 调试：打印权重指针（十六进制格式）
        import sys
        print(f"Weights pointers:", flush=True)
        print(f"  in_embed  = {hex(self._weights.in_embed) if self._weights.in_embed else 'NULL'}", flush=True)
        print(f"  out_embed = {hex(self._weights.out_embed) if self._weights.out_embed else 'NULL'}", flush=True)
        print(f"  out_norm_w = {hex(self._weights.out_norm_w) if self._weights.out_norm_w else 'NULL'}", flush=True)
        sys.stdout.flush()
        
        # 检查指针有效性
        if not self._weights.in_embed or not self._weights.out_embed or not self._weights.out_norm_w:
            raise RuntimeError("One or more weight tensors are NULL!")
        
        # 5. 加载权重
        print("Loading weights...", flush=True)
        self._load_weights(model_path)
        print("Model loaded successfully!", flush=True)
    
    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
    
    def _load_weights(self, model_path):
        """从 safetensors 文件加载权重"""
        import torch
        
        # 收集所有权重名称，检查是否有 tied embeddings
        all_weight_names = set()
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(str(file), framework="pt", device="cpu") as f:
                all_weight_names.update(f.keys())
        
        # 检查是否需要复制 embed_tokens 到 lm_head（tied embeddings）
        has_lm_head = "lm_head.weight" in all_weight_names
        
        for file in sorted(model_path.glob("*.safetensors")):
            # 使用 PyTorch 框架读取，因为 NumPy 不支持 bfloat16
            with safetensors.safe_open(str(file), framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor_data = f.get_tensor(name)
                    self._load_single_weight(name, tensor_data, has_lm_head)
    
    def _load_single_weight(self, name: str, data, has_lm_head: bool = True):
        """将单个权重加载到对应的张量"""
        import ctypes
        
        # PyTorch 张量需要先确保是连续的，然后获取数据指针
        if not data.is_contiguous():
            data = data.contiguous()
        
        # 获取 PyTorch 张量的数据指针
        data_ptr = ctypes.c_void_p(data.data_ptr())
        
        def safe_tensor_load(tensor_ptr, data_ptr, tensor_name):
            """安全地调用 tensorLoad，带有详细的错误信息"""
            if not tensor_ptr:
                raise RuntimeError(f"Tensor pointer for '{tensor_name}' is NULL!")
            print(f"    tensorLoad({tensor_name}={hex(tensor_ptr)}, data_ptr={hex(data_ptr.value)})", flush=True)
            LIB_LLAISYS.tensorLoad(tensor_ptr, data_ptr)
        
        # 根据名称匹配到对应的张量
        if name == "model.embed_tokens.weight":
            print(f"  Loading {name}: shape={list(data.shape)}, dtype={data.dtype}", flush=True)
            safe_tensor_load(self._weights.in_embed, data_ptr, "in_embed")
            # 如果没有单独的 lm_head，则复制 embed_tokens 到 lm_head (tied embeddings)
            if not has_lm_head:
                print(f"  Copying embed_tokens to lm_head (tied embeddings)", flush=True)
                safe_tensor_load(self._weights.out_embed, data_ptr, "out_embed")
        elif name == "lm_head.weight":
            print(f"  Loading {name}: shape={list(data.shape)}, dtype={data.dtype}", flush=True)
            safe_tensor_load(self._weights.out_embed, data_ptr, "out_embed")
        elif name == "model.norm.weight":
            print(f"  Loading {name}: shape={list(data.shape)}, dtype={data.dtype}", flush=True)
            safe_tensor_load(self._weights.out_norm_w, data_ptr, "out_norm_w")
        else:
            # 解析层权重
            self._load_layer_weight(name, data, data_ptr)
    
    def _load_layer_weight(self, name: str, data, data_ptr):
        """加载层权重"""
        import re
        
        # 匹配模式: model.layers.{layer_idx}.{component}.{param}
        match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        if not match:
            print(f"Warning: Unknown weight name: {name}")
            return
        
        layer_idx = int(match.group(1))
        component_name = match.group(2)
        
        weight_mapping = {
            "input_layernorm.weight": ("attn_norm_w", layer_idx),
            "self_attn.q_proj.weight": ("attn_q_w", layer_idx),
            "self_attn.q_proj.bias": ("attn_q_b", layer_idx),
            "self_attn.k_proj.weight": ("attn_k_w", layer_idx),
            "self_attn.k_proj.bias": ("attn_k_b", layer_idx),
            "self_attn.v_proj.weight": ("attn_v_w", layer_idx),
            "self_attn.v_proj.bias": ("attn_v_b", layer_idx),
            "self_attn.o_proj.weight": ("attn_o_w", layer_idx),
            "post_attention_layernorm.weight": ("mlp_norm_w", layer_idx),
            "mlp.gate_proj.weight": ("mlp_gate_w", layer_idx),
            "mlp.up_proj.weight": ("mlp_up_w", layer_idx),
            "mlp.down_proj.weight": ("mlp_down_w", layer_idx),
        }
        
        if component_name in weight_mapping:
            attr_name, idx = weight_mapping[component_name]
            tensor = getattr(self._weights, attr_name)[idx]
            LIB_LLAISYS.tensorLoad(tensor, data_ptr)
        else:
            print(f"Warning: Unknown component: {component_name}")
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """生成文本序列"""
        if max_new_tokens is None:
            max_new_tokens = 128
        
        # 重置模型状态
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        
        # 转换输入为 ctypes 数组
        tokens = list(inputs)
        
        # 第一次推理（处理整个 prompt）
        token_array = (c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, len(tokens))
        
        if next_token < 0:
            raise RuntimeError("Inference failed")
        
        tokens.append(next_token)
        
        # 自回归生成
        for _ in range(max_new_tokens - 1):
            # 只输入最后一个 token（使用 KV-Cache）
            token_array = (c_int64 * 1)(tokens[-1])
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, 1)
            
            if next_token < 0:
                raise RuntimeError("Inference failed")
            
            tokens.append(next_token)
            
            # 检查是否到达结束 token
            if next_token == self._meta.end_token:
                break
        
        return tokens
