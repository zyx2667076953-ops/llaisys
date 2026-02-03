#include "qwen2_model.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Config& cfg) 
    : config_(cfg), current_pos_(0) {
    
    // allocate weight tensors
    embed_tokens_ = Tensor::create({cfg.voc, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
    lm_head_ = Tensor::create({cfg.voc, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
    final_norm_w_ = Tensor::create({cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
    
    attn_norm_w_.resize(cfg.nlayer);
    q_proj_w_.resize(cfg.nlayer);
    q_proj_b_.resize(cfg.nlayer);
    k_proj_w_.resize(cfg.nlayer);
    k_proj_b_.resize(cfg.nlayer);
    v_proj_w_.resize(cfg.nlayer);
    v_proj_b_.resize(cfg.nlayer);
    o_proj_w_.resize(cfg.nlayer);
    mlp_norm_w_.resize(cfg.nlayer);
    gate_proj_w_.resize(cfg.nlayer);
    up_proj_w_.resize(cfg.nlayer);
    down_proj_w_.resize(cfg.nlayer);
    
    for (size_t i = 0; i < cfg.nlayer; ++i) {
        attn_norm_w_[i] = Tensor::create({cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        q_proj_w_[i] = Tensor::create({cfg.nh * cfg.dh, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        q_proj_b_[i] = Tensor::create({cfg.nh * cfg.dh}, cfg.dtype, cfg.device_type, cfg.device_id);
        k_proj_w_[i] = Tensor::create({cfg.nkvh * cfg.dh, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        k_proj_b_[i] = Tensor::create({cfg.nkvh * cfg.dh}, cfg.dtype, cfg.device_type, cfg.device_id);
        v_proj_w_[i] = Tensor::create({cfg.nkvh * cfg.dh, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        v_proj_b_[i] = Tensor::create({cfg.nkvh * cfg.dh}, cfg.dtype, cfg.device_type, cfg.device_id);
        o_proj_w_[i] = Tensor::create({cfg.hs, cfg.nh * cfg.dh}, cfg.dtype, cfg.device_type, cfg.device_id);
        
        mlp_norm_w_[i] = Tensor::create({cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        gate_proj_w_[i] = Tensor::create({cfg.di, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        up_proj_w_[i] = Tensor::create({cfg.di, cfg.hs}, cfg.dtype, cfg.device_type, cfg.device_id);
        down_proj_w_[i] = Tensor::create({cfg.hs, cfg.di}, cfg.dtype, cfg.device_type, cfg.device_id);
    }
    
    // init kv cache
    kv_cache_ = std::make_unique<ModelKVCache>(
        cfg.nlayer, cfg.maxseq, cfg.nkvh, cfg.dh, cfg.dtype, cfg.device_type, cfg.device_id
    );
}

tensor_t Qwen2Model::create_position_ids(size_t len) {
    auto pos = Tensor::create({len}, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
    
    if (config_.device_type == LLAISYS_DEVICE_CPU) {
        int64_t* p = reinterpret_cast<int64_t*>(pos->data());
        for (size_t i = 0; i < len; ++i) p[i] = current_pos_ + i;
    } else {
        std::vector<int64_t> tmp(len);
        for (size_t i = 0; i < len; ++i) tmp[i] = current_pos_ + i;
        pos->load(tmp.data());
    }
    return pos;
}

tensor_t Qwen2Model::transformer_layer(tensor_t hidden, size_t layer, tensor_t pos) {
    size_t seq = hidden->shape()[0];
    size_t hs = config_.hs;
    size_t nh = config_.nh;
    size_t nkvh = config_.nkvh;
    size_t dh = config_.dh;
    size_t di = config_.di;
    
    // attention
    auto normed = Tensor::create({seq, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::rms_norm(normed, hidden, attn_norm_w_[layer], config_.epsilon);
    
    auto q = Tensor::create({seq, nh * dh}, config_.dtype, config_.device_type, config_.device_id);
    auto k = Tensor::create({seq, nkvh * dh}, config_.dtype, config_.device_type, config_.device_id);
    auto v = Tensor::create({seq, nkvh * dh}, config_.dtype, config_.device_type, config_.device_id);
    
    ops::linear(q, normed, q_proj_w_[layer], q_proj_b_[layer]);
    ops::linear(k, normed, k_proj_w_[layer], k_proj_b_[layer]);
    ops::linear(v, normed, v_proj_w_[layer], v_proj_b_[layer]);
    
    q = q->view({seq, nh, dh});
    k = k->view({seq, nkvh, dh});
    v = v->view({seq, nkvh, dh});
    
    // rope
    auto qr = Tensor::create(q->shape(), config_.dtype, config_.device_type, config_.device_id);
    auto kr = Tensor::create(k->shape(), config_.dtype, config_.device_type, config_.device_id);
    ops::rope(qr, q, pos, config_.theta);
    ops::rope(kr, k, pos, config_.theta);
    
    // kv cache
    auto& cache = kv_cache_->get_layer(layer);
    auto fk = cache.update_k(kr);
    auto fv = cache.update_v(v);
    (void)cache.current_len;
    
    // attention
    float scale = 1.0f / std::sqrt((float)dh);
    auto attn = Tensor::create({seq, nh, dh}, config_.dtype, config_.device_type, config_.device_id);
    ops::self_attention(attn, qr, fk, fv, scale);
    
    attn = attn->view({seq, nh * dh});
    auto attn_out = Tensor::create({seq, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(attn_out, attn, o_proj_w_[layer], nullptr);
    
    // residual
    auto h1 = Tensor::create({seq, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::add(h1, hidden, attn_out);
    
    // mlp
    auto mlp_in = Tensor::create({seq, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::rms_norm(mlp_in, h1, mlp_norm_w_[layer], config_.epsilon);
    
    auto gate = Tensor::create({seq, di}, config_.dtype, config_.device_type, config_.device_id);
    auto up = Tensor::create({seq, di}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(gate, mlp_in, gate_proj_w_[layer], nullptr);
    ops::linear(up, mlp_in, up_proj_w_[layer], nullptr);
    
    auto act = Tensor::create({seq, di}, config_.dtype, config_.device_type, config_.device_id);
    ops::swiglu(act, gate, up);
    
    auto mlp_out = Tensor::create({seq, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(mlp_out, act, down_proj_w_[layer], nullptr);
    
    auto out = Tensor::create({seq, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::add(out, h1, mlp_out);
    
    return out;
}

tensor_t Qwen2Model::forward(tensor_t input_ids) {
    size_t seq = input_ids->numel();
    
    auto hidden = Tensor::create({seq, config_.hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::embedding(hidden, input_ids, embed_tokens_);
    
    auto pos = create_position_ids(seq);
    
    for (size_t l = 0; l < config_.nlayer; ++l) {
        hidden = transformer_layer(hidden, l, pos);
    }
    
    auto normed = Tensor::create({seq, config_.hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::rms_norm(normed, hidden, final_norm_w_, config_.epsilon);
    
    auto logits = Tensor::create({seq, config_.voc}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(logits, normed, lm_head_, nullptr);
    
    return logits;
}

int64_t Qwen2Model::infer_one_step(const int64_t* tokens, size_t n) {
    auto ids = Tensor::create({n}, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
    ids->load(tokens);
    
    auto logits = forward(ids);
    
    auto last = logits->slice(0, n - 1, n);
    last = last->view({config_.voc});
    
    auto idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    auto val = Tensor::create({1}, config_.dtype, LLAISYS_DEVICE_CPU, 0);
    
    ASSERT(last->deviceType() == LLAISYS_DEVICE_CPU, "only cpu inference supported");
    
    ops::argmax(idx, val, last);
    
    int64_t next = *reinterpret_cast<int64_t*>(idx->data());
    current_pos_ += n;
    
    return next;
}

void Qwen2Model::reset() {
    current_pos_ = 0;
    kv_cache_->reset_all();
}

} // namespace llaisys::models
