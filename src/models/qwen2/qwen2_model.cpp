#include "qwen2_model.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Config& config) 
    : config_(config), current_pos_(0) {
    
    // 初始化权重张量（只分配空间，数据从外部加载）
    embed_tokens_ = Tensor::create({config.voc, config.hs}, config.dtype, config.device_type, config.device_id);
    lm_head_ = Tensor::create({config.voc, config.hs}, config.dtype, config.device_type, config.device_id);
    final_norm_w_ = Tensor::create({config.hs}, config.dtype, config.device_type, config.device_id);
    
    // 为每一层分配权重
    attn_norm_w_.resize(config.nlayer);
    q_proj_w_.resize(config.nlayer);
    q_proj_b_.resize(config.nlayer);
    k_proj_w_.resize(config.nlayer);
    k_proj_b_.resize(config.nlayer);
    v_proj_w_.resize(config.nlayer);
    v_proj_b_.resize(config.nlayer);
    o_proj_w_.resize(config.nlayer);
    mlp_norm_w_.resize(config.nlayer);
    gate_proj_w_.resize(config.nlayer);
    up_proj_w_.resize(config.nlayer);
    down_proj_w_.resize(config.nlayer);
    
    for (size_t i = 0; i < config.nlayer; ++i) {
        attn_norm_w_[i] = Tensor::create({config.hs}, config.dtype, config.device_type, config.device_id);
        q_proj_w_[i] = Tensor::create({config.nh * config.dh, config.hs}, config.dtype, config.device_type, config.device_id);
        q_proj_b_[i] = Tensor::create({config.nh * config.dh}, config.dtype, config.device_type, config.device_id);
        k_proj_w_[i] = Tensor::create({config.nkvh * config.dh, config.hs}, config.dtype, config.device_type, config.device_id);
        k_proj_b_[i] = Tensor::create({config.nkvh * config.dh}, config.dtype, config.device_type, config.device_id);
        v_proj_w_[i] = Tensor::create({config.nkvh * config.dh, config.hs}, config.dtype, config.device_type, config.device_id);
        v_proj_b_[i] = Tensor::create({config.nkvh * config.dh}, config.dtype, config.device_type, config.device_id);
        o_proj_w_[i] = Tensor::create({config.hs, config.nh * config.dh}, config.dtype, config.device_type, config.device_id);
        
        mlp_norm_w_[i] = Tensor::create({config.hs}, config.dtype, config.device_type, config.device_id);
        gate_proj_w_[i] = Tensor::create({config.di, config.hs}, config.dtype, config.device_type, config.device_id);
        up_proj_w_[i] = Tensor::create({config.di, config.hs}, config.dtype, config.device_type, config.device_id);
        down_proj_w_[i] = Tensor::create({config.hs, config.di}, config.dtype, config.device_type, config.device_id);
    }
    
    // 初始化 KV-Cache
    kv_cache_ = std::make_unique<ModelKVCache>(
        config.nlayer, config.maxseq, config.nkvh, config.dh,
        config.dtype, config.device_type, config.device_id
    );
}

tensor_t Qwen2Model::create_position_ids(size_t seqlen) {
    auto pos_ids = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
    
    if (config_.device_type == LLAISYS_DEVICE_CPU) {
        int64_t* data = reinterpret_cast<int64_t*>(pos_ids->data());
        for (size_t i = 0; i < seqlen; ++i) {
            data[i] = current_pos_ + i;
        }
    } else {
        // GPU: 在 CPU 创建，然后拷贝
        std::vector<int64_t> pos_data(seqlen);
        for (size_t i = 0; i < seqlen; ++i) {
            pos_data[i] = current_pos_ + i;
        }
        pos_ids->load(pos_data.data());
    }
    
    return pos_ids;
}

tensor_t Qwen2Model::transformer_layer(tensor_t hidden, size_t layer_idx, tensor_t pos_ids) {
    size_t seqlen = hidden->shape()[0];
    size_t hs = config_.hs;
    size_t nh = config_.nh;
    size_t nkvh = config_.nkvh;
    size_t dh = config_.dh;
    size_t di = config_.di;
    
    // === Self-Attention 分支 ===
    
    // 1. RMS Norm
    auto attn_normed = Tensor::create({seqlen, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::rms_norm(attn_normed, hidden, attn_norm_w_[layer_idx], config_.epsilon);
    
    // 2. Q, K, V 投影
    auto q = Tensor::create({seqlen, nh * dh}, config_.dtype, config_.device_type, config_.device_id);
    auto k = Tensor::create({seqlen, nkvh * dh}, config_.dtype, config_.device_type, config_.device_id);
    auto v = Tensor::create({seqlen, nkvh * dh}, config_.dtype, config_.device_type, config_.device_id);
    
    ops::linear(q, attn_normed, q_proj_w_[layer_idx], q_proj_b_[layer_idx]);
    ops::linear(k, attn_normed, k_proj_w_[layer_idx], k_proj_b_[layer_idx]);
    ops::linear(v, attn_normed, v_proj_w_[layer_idx], v_proj_b_[layer_idx]);
    
    // 3. Reshape to [seqlen, nhead, dh]
    q = q->view({seqlen, nh, dh});
    k = k->view({seqlen, nkvh, dh});
    v = v->view({seqlen, nkvh, dh});
    
    // 4. RoPE
    auto q_rope = Tensor::create(q->shape(), config_.dtype, config_.device_type, config_.device_id);
    auto k_rope = Tensor::create(k->shape(), config_.dtype, config_.device_type, config_.device_id);
    ops::rope(q_rope, q, pos_ids, config_.theta);
    ops::rope(k_rope, k, pos_ids, config_.theta);
    
    // 5. 更新 KV-Cache
    auto& layer_cache = kv_cache_->get_layer(layer_idx);
    auto full_k = layer_cache.update_k(k_rope);
    auto full_v = layer_cache.update_v(v);
    size_t total_len = layer_cache.current_len;
    
    // 6. Self-Attention
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    auto attn_out = Tensor::create({seqlen, nh, dh}, config_.dtype, config_.device_type, config_.device_id);
    ops::self_attention(attn_out, q_rope, full_k, full_v, scale);
    
    // 7. Reshape back to [seqlen, nh*dh]
    attn_out = attn_out->view({seqlen, nh * dh});
    
    // 8. 输出投影
    auto attn_output = Tensor::create({seqlen, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(attn_output, attn_out, o_proj_w_[layer_idx], nullptr);
    
    // 9. Residual
    auto after_attn = Tensor::create({seqlen, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::add(after_attn, hidden, attn_output);
    
    // === MLP 分支 ===
    
    // 10. RMS Norm
    auto mlp_normed = Tensor::create({seqlen, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::rms_norm(mlp_normed, after_attn, mlp_norm_w_[layer_idx], config_.epsilon);
    
    // 11. Gate 和 Up 投影
    auto gate = Tensor::create({seqlen, di}, config_.dtype, config_.device_type, config_.device_id);
    auto up = Tensor::create({seqlen, di}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(gate, mlp_normed, gate_proj_w_[layer_idx], nullptr);
    ops::linear(up, mlp_normed, up_proj_w_[layer_idx], nullptr);
    
    // 12. SwiGLU
    auto activated = Tensor::create({seqlen, di}, config_.dtype, config_.device_type, config_.device_id);
    ops::swiglu(activated, gate, up);
    
    // 13. Down 投影
    auto mlp_output = Tensor::create({seqlen, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(mlp_output, activated, down_proj_w_[layer_idx], nullptr);
    
    // 14. Residual
    auto layer_output = Tensor::create({seqlen, hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::add(layer_output, after_attn, mlp_output);
    
    return layer_output;
}

tensor_t Qwen2Model::forward(tensor_t input_ids) {
    size_t seqlen = input_ids->numel();
    
    // 1. Embedding
    auto hidden = Tensor::create({seqlen, config_.hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::embedding(hidden, input_ids, embed_tokens_);
    
    // 2. 创建位置 IDs
    auto pos_ids = create_position_ids(seqlen);
    
    // 3. 逐层 Transformer
    for (size_t layer = 0; layer < config_.nlayer; ++layer) {
        hidden = transformer_layer(hidden, layer, pos_ids);
    }
    
    // 4. 最终归一化
    auto normed = Tensor::create({seqlen, config_.hs}, config_.dtype, config_.device_type, config_.device_id);
    ops::rms_norm(normed, hidden, final_norm_w_, config_.epsilon);
    
    // 5. LM Head
    auto logits = Tensor::create({seqlen, config_.voc}, config_.dtype, config_.device_type, config_.device_id);
    ops::linear(logits, normed, lm_head_, nullptr);
    
    return logits;
}

int64_t Qwen2Model::infer_one_step(const int64_t* token_ids, size_t ntoken) {
    // 1. 创建输入张量
    auto input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
    input_ids->load(token_ids);
    
    // 2. 前向传播
    auto logits = forward(input_ids);  // [ntoken, voc]
    
    // 3. 取最后一个 token 的 logits
    auto last_logits = logits->slice(0, ntoken - 1, ntoken);  // [1, voc]
    last_logits = last_logits->view({config_.voc});           // [voc]
    
    // 4. Argmax
    // 注意：argmax 的输出张量在 CPU 上创建
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    auto max_val = Tensor::create({1}, config_.dtype, LLAISYS_DEVICE_CPU, 0);
    
    // 目前只支持 CPU 推理，GPU 支持需要实现 tensor->to()
    ASSERT(last_logits->deviceType() == LLAISYS_DEVICE_CPU, 
           "GPU inference not yet supported, please use CPU device");
    
    ops::argmax(max_idx, max_val, last_logits);
    
    // 5. 读取结果
    int64_t next_token = *reinterpret_cast<int64_t*>(max_idx->data());
    
    // 6. 更新位置
    current_pos_ += ntoken;
    
    return next_token;
}

void Qwen2Model::reset() {
    current_pos_ = 0;
    kv_cache_->reset_all();
}

} // namespace llaisys::models
