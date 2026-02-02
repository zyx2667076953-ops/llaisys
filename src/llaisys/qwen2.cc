#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2_model.hpp"
#include "../utils.hpp"
#include "llaisys_tensor.hpp"
#include <iostream>
#include <vector>

using namespace llaisys;
using namespace llaisys::models;

struct LlaisysQwen2Model {
    Qwen2Model* model;
    LlaisysQwen2Weights weights;
    
    // 用于存储 LlaisysTensor 包装器的生命周期
    std::vector<LlaisysTensor*> tensor_wrappers;
    
    ~LlaisysQwen2Model() {
        // 释放所有包装器
        for (auto* wrapper : tensor_wrappers) {
            delete wrapper;
        }
    }
};

// 辅助函数：将 tensor_t 包装成 llaisysTensor_t
static llaisysTensor_t wrap_tensor(tensor_t t, std::vector<LlaisysTensor*>& wrappers) {
    auto* wrapper = new LlaisysTensor{t};
    wrappers.push_back(wrapper);
    return wrapper;
}

__C __export struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice
) {
    try {
        // 转换 meta 到 config
        Qwen2Config config;
        config.nlayer = meta->nlayer;
        config.hs = meta->hs;
        config.nh = meta->nh;
        config.nkvh = meta->nkvh;
        config.dh = meta->dh;
        config.di = meta->di;
        config.maxseq = meta->maxseq;
        config.voc = meta->voc;
        config.epsilon = meta->epsilon;
        config.theta = meta->theta;
        config.eos_token_id = meta->end_token;
        config.dtype = meta->dtype;
        config.device_type = device;
        config.device_id = (ndevice > 0) ? device_ids[0] : 0;
        
        // 创建模型
        auto* cpp_model = new Qwen2Model(config);
        
        // 创建包装器
        auto* model = new LlaisysQwen2Model();
        model->model = cpp_model;
        
        // 填充权重结构（使用正确的 LlaisysTensor 包装器）
        model->weights.in_embed = wrap_tensor(cpp_model->embed_tokens(), model->tensor_wrappers);
        model->weights.out_embed = wrap_tensor(cpp_model->lm_head(), model->tensor_wrappers);
        model->weights.out_norm_w = wrap_tensor(cpp_model->final_norm_w(), model->tensor_wrappers);
        
        // 分配层权重数组
        model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];
        
        for (size_t i = 0; i < meta->nlayer; ++i) {
            model->weights.attn_norm_w[i] = wrap_tensor(cpp_model->attn_norm_w(i), model->tensor_wrappers);
            model->weights.attn_q_w[i] = wrap_tensor(cpp_model->q_proj_w(i), model->tensor_wrappers);
            model->weights.attn_q_b[i] = wrap_tensor(cpp_model->q_proj_b(i), model->tensor_wrappers);
            model->weights.attn_k_w[i] = wrap_tensor(cpp_model->k_proj_w(i), model->tensor_wrappers);
            model->weights.attn_k_b[i] = wrap_tensor(cpp_model->k_proj_b(i), model->tensor_wrappers);
            model->weights.attn_v_w[i] = wrap_tensor(cpp_model->v_proj_w(i), model->tensor_wrappers);
            model->weights.attn_v_b[i] = wrap_tensor(cpp_model->v_proj_b(i), model->tensor_wrappers);
            model->weights.attn_o_w[i] = wrap_tensor(cpp_model->o_proj_w(i), model->tensor_wrappers);
            model->weights.mlp_norm_w[i] = wrap_tensor(cpp_model->mlp_norm_w(i), model->tensor_wrappers);
            model->weights.mlp_gate_w[i] = wrap_tensor(cpp_model->gate_proj_w(i), model->tensor_wrappers);
            model->weights.mlp_up_w[i] = wrap_tensor(cpp_model->up_proj_w(i), model->tensor_wrappers);
            model->weights.mlp_down_w[i] = wrap_tensor(cpp_model->down_proj_w(i), model->tensor_wrappers);
        }
        
        return model;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to create Qwen2 model: " << e.what() << std::endl;
        return nullptr;
    }
}

__C __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model) {
    if (model) {
        delete[] model->weights.attn_norm_w;
        delete[] model->weights.attn_q_w;
        delete[] model->weights.attn_q_b;
        delete[] model->weights.attn_k_w;
        delete[] model->weights.attn_k_b;
        delete[] model->weights.attn_v_w;
        delete[] model->weights.attn_v_b;
        delete[] model->weights.attn_o_w;
        delete[] model->weights.mlp_norm_w;
        delete[] model->weights.mlp_gate_w;
        delete[] model->weights.mlp_up_w;
        delete[] model->weights.mlp_down_w;
        
        delete model->model;
        delete model;
    }
}

__C __export struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model) {
    if (!model) return nullptr;
    return &(model->weights);
}

__C __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    if (!model || !token_ids) return -1;
    
    try {
        return model->model->infer_one_step(token_ids, ntoken);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Qwen2 inference failed: " << e.what() << std::endl;
        return -1;
    }
}

__C __export void llaisysQwen2ModelReset(struct LlaisysQwen2Model* model) {
    if (model) {
        model->model->reset();
    }
}
