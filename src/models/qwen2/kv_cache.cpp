#include "kv_cache.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::models {

tensor_t LayerKVCache::update_k(tensor_t new_k) {
    if (!k_cache) {
        return new_k;
    }
    
    size_t new_len = new_k->shape()[0];
    size_t nkvhead = new_k->shape()[1];
    size_t dh = new_k->shape()[2];
    
    // 将 new_k 复制到 k_cache 的相应位置
    // k_cache: [max_len, nkvhead, dh]
    // 需要复制到 [current_len:current_len+new_len, :, :]
    
    size_t elem_size = new_k->elementSize();
    size_t row_size = nkvhead * dh * elem_size;
    
    std::byte* cache_data = k_cache->data();
    const std::byte* new_data = new_k->data();
    
    // 保存旧的长度用于复制（不在这里更新 current_len）
    size_t old_len = current_len;
    
    for (size_t i = 0; i < new_len; ++i) {
        std::memcpy(cache_data + (old_len + i) * row_size,
                    new_data + i * row_size,
                    row_size);
    }
    
    // 注意：不在这里更新 current_len，等 update_v 完成后再更新
    // 返回切片 [0:old_len+new_len, :, :]
    return k_cache->slice(0, 0, old_len + new_len);
}

tensor_t LayerKVCache::update_v(tensor_t new_v) {
    if (!v_cache) {
        return new_v;
    }
    
    size_t new_len = new_v->shape()[0];
    size_t nkvhead = new_v->shape()[1];
    size_t dh = new_v->shape()[2];
    
    size_t elem_size = new_v->elementSize();
    size_t row_size = nkvhead * dh * elem_size;
    
    std::byte* cache_data = v_cache->data();
    const std::byte* new_data = new_v->data();
    
    // 使用当前的 current_len（update_k 没有修改它）
    size_t old_len = current_len;
    
    for (size_t i = 0; i < new_len; ++i) {
        std::memcpy(cache_data + (old_len + i) * row_size,
                    new_data + i * row_size,
                    row_size);
    }
    
    // 现在更新 current_len（K 和 V 都已复制完成）
    current_len += new_len;
    
    // 返回切片 [0:current_len, :, :]
    return v_cache->slice(0, 0, current_len);
}

ModelKVCache::ModelKVCache(size_t nlayer, size_t max_len, size_t nkvhead, size_t dh,
                           llaisysDataType_t dtype, llaisysDeviceType_t device_type, int device_id)
    : _max_len(max_len), _nlayer(nlayer), _nkvhead(nkvhead), _dh(dh),
      _dtype(dtype), _device_type(device_type), _device_id(device_id) {
    
    _layers.resize(nlayer);
    
    for (size_t i = 0; i < nlayer; ++i) {
        _layers[i].k_cache = Tensor::create({max_len, nkvhead, dh}, dtype, device_type, device_id);
        _layers[i].v_cache = Tensor::create({max_len, nkvhead, dh}, dtype, device_type, device_id);
        _layers[i].current_len = 0;
    }
}

void ModelKVCache::reset_all() {
    for (auto& layer : _layers) {
        layer.reset();
    }
}

} // namespace llaisys::models
