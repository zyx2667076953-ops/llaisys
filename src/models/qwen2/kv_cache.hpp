#pragma once
#include "../../tensor/tensor.hpp"
#include <vector>

namespace llaisys::models {

// 单层的 KV-Cache
struct LayerKVCache {
    tensor_t k_cache;  // [max_len, nkvhead, dh]
    tensor_t v_cache;  // [max_len, nkvhead, dh]
    size_t current_len;  // 当前已使用的长度
    
    LayerKVCache() : current_len(0) {}
    
    // 重置缓存
    void reset() {
        current_len = 0;
    }
    
    // 更新缓存并返回完整的 K/V
    tensor_t update_k(tensor_t new_k);  // new_k: [seqlen, nkvhead, dh]
    tensor_t update_v(tensor_t new_v);  // new_v: [seqlen, nkvhead, dh]
};

// 整个模型的 KV-Cache
class ModelKVCache {
private:
    std::vector<LayerKVCache> _layers;
    size_t _max_len;
    size_t _nlayer;
    size_t _nkvhead;
    size_t _dh;
    llaisysDataType_t _dtype;
    llaisysDeviceType_t _device_type;
    int _device_id;

public:
    ModelKVCache(size_t nlayer, size_t max_len, size_t nkvhead, size_t dh,
                 llaisysDataType_t dtype, llaisysDeviceType_t device_type, int device_id);
    
    LayerKVCache& get_layer(size_t layer_idx) { return _layers[layer_idx]; }
    
    void reset_all();
    size_t max_len() const { return _max_len; }
};

} // namespace llaisys::models
