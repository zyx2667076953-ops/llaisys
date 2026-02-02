#pragma once
#include "../../tensor/tensor.hpp"
#include "../../ops/ops.hpp"
#include "kv_cache.hpp"
#include <vector>
#include <memory>
#include <string>

namespace llaisys::models {

struct Qwen2Config {
    size_t nlayer;
    size_t hs;         // hidden size
    size_t nh;         // num attention heads
    size_t nkvh;       // num kv heads
    size_t dh;         // head dim
    size_t di;         // intermediate size (MLP)
    size_t maxseq;     // max sequence length
    size_t voc;        // vocab size
    float epsilon;
    float theta;
    int64_t eos_token_id;
    llaisysDataType_t dtype;
    llaisysDeviceType_t device_type;
    int device_id;
};

class Qwen2Model {
private:
    Qwen2Config config_;
    
    // 权重
    tensor_t embed_tokens_;
    tensor_t lm_head_;
    tensor_t final_norm_w_;
    
    std::vector<tensor_t> attn_norm_w_;
    std::vector<tensor_t> q_proj_w_;
    std::vector<tensor_t> q_proj_b_;
    std::vector<tensor_t> k_proj_w_;
    std::vector<tensor_t> k_proj_b_;
    std::vector<tensor_t> v_proj_w_;
    std::vector<tensor_t> v_proj_b_;
    std::vector<tensor_t> o_proj_w_;
    
    std::vector<tensor_t> mlp_norm_w_;
    std::vector<tensor_t> gate_proj_w_;
    std::vector<tensor_t> up_proj_w_;
    std::vector<tensor_t> down_proj_w_;
    
    // KV-Cache
    std::unique_ptr<ModelKVCache> kv_cache_;
    size_t current_pos_;

public:
    Qwen2Model(const Qwen2Config& config);
    
    // 获取权重指针
    tensor_t& embed_tokens() { return embed_tokens_; }
    tensor_t& lm_head() { return lm_head_; }
    tensor_t& final_norm_w() { return final_norm_w_; }
    tensor_t& attn_norm_w(size_t i) { return attn_norm_w_[i]; }
    tensor_t& q_proj_w(size_t i) { return q_proj_w_[i]; }
    tensor_t& q_proj_b(size_t i) { return q_proj_b_[i]; }
    tensor_t& k_proj_w(size_t i) { return k_proj_w_[i]; }
    tensor_t& k_proj_b(size_t i) { return k_proj_b_[i]; }
    tensor_t& v_proj_w(size_t i) { return v_proj_w_[i]; }
    tensor_t& v_proj_b(size_t i) { return v_proj_b_[i]; }
    tensor_t& o_proj_w(size_t i) { return o_proj_w_[i]; }
    tensor_t& mlp_norm_w(size_t i) { return mlp_norm_w_[i]; }
    tensor_t& gate_proj_w(size_t i) { return gate_proj_w_[i]; }
    tensor_t& up_proj_w(size_t i) { return up_proj_w_[i]; }
    tensor_t& down_proj_w(size_t i) { return down_proj_w_[i]; }
    
    // 前向传播
    tensor_t forward(tensor_t input_ids);
    
    // 推理一步
    int64_t infer_one_step(const int64_t* token_ids, size_t ntoken);
    
    // 重置
    void reset();

private:
    tensor_t transformer_layer(tensor_t hidden, size_t layer_idx, tensor_t pos_ids);
    tensor_t create_position_ids(size_t seqlen);
};

} // namespace llaisys::models
