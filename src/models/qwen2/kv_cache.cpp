#include "kv_cache.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::models {

tensor_t LayerKVCache::update_k(tensor_t new_k) {
    if (!k_cache) return new_k;
    
    size_t len = new_k->shape()[0];
    size_t nkvh = new_k->shape()[1];
    size_t dh = new_k->shape()[2];
    
    size_t row_bytes = nkvh * dh * new_k->elementSize();
    std::byte* dst = k_cache->data();
    const std::byte* src = new_k->data();
    
    for (size_t i = 0; i < len; ++i) {
        std::memcpy(dst + (current_len + i) * row_bytes, src + i * row_bytes, row_bytes);
    }
    
    return k_cache->slice(0, 0, current_len + len);
}

tensor_t LayerKVCache::update_v(tensor_t new_v) {
    if (!v_cache) return new_v;
    
    size_t len = new_v->shape()[0];
    size_t nkvh = new_v->shape()[1];
    size_t dh = new_v->shape()[2];
    
    size_t row_bytes = nkvh * dh * new_v->elementSize();
    std::byte* dst = v_cache->data();
    const std::byte* src = new_v->data();
    
    for (size_t i = 0; i < len; ++i) {
        std::memcpy(dst + (current_len + i) * row_bytes, src + i * row_bytes, row_bytes);
    }
    
    current_len += len;
    return v_cache->slice(0, 0, current_len);
}

ModelKVCache::ModelKVCache(size_t nlayer, size_t maxlen, size_t nkvh, size_t dh,
                           llaisysDataType_t dtype, llaisysDeviceType_t dev, int dev_id)
    : _max_len(maxlen), _nlayer(nlayer), _nkvhead(nkvh), _dh(dh),
      _dtype(dtype), _device_type(dev), _device_id(dev_id) {
    
    _layers.resize(nlayer);
    for (size_t i = 0; i < nlayer; ++i) {
        _layers[i].k_cache = Tensor::create({maxlen, nkvh, dh}, dtype, dev, dev_id);
        _layers[i].v_cache = Tensor::create({maxlen, nkvh, dh}, dtype, dev, dev_id);
        _layers[i].current_len = 0;
    }
}

void ModelKVCache::reset_all() {
    for (auto& layer : _layers) {
        layer.reset();
    }
}

} // namespace llaisys::models
