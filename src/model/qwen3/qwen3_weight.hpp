// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "model/common/weights.hpp"

namespace powerserve {

struct Qwen3LayerWeights : LayerWeights {

public:
    Qwen3LayerWeights(ggml_context *ctx, uint32_t layer) {
        attn_norm   = get_tensor(ctx, layer, "attn_norm.weight");
        ffn_norm    = get_tensor(ctx, layer, "ffn_norm.weight");
        attn_q      = get_tensor(ctx, layer, "attn_q.weight");
        attn_q_norm = get_tensor(ctx, layer, "attn_q_norm.weight");
        attn_k      = get_tensor(ctx, layer, "attn_k.weight");
        attn_k_norm = get_tensor(ctx, layer, "attn_k_norm.weight");
        attn_v      = get_tensor(ctx, layer, "attn_v.weight");
        attn_output = get_tensor(ctx, layer, "attn_output.weight");
        ffn_gate    = get_tensor(ctx, layer, "ffn_gate.weight");
        ffn_up      = get_tensor(ctx, layer, "ffn_up.weight");
        ffn_down    = get_tensor(ctx, layer, "ffn_down.weight");
    }

    ~Qwen3LayerWeights() override = default;
};

struct Qwen3Weight : Weight {

public:
    Qwen3Weight(ggml_context *ctx, uint32_t n_layers, bool lazy_load) : Weight(ctx, lazy_load) {
        if (!lazy_load) {
            for (size_t layer = 0; layer < n_layers; layer++) {
                lw.push_back(Qwen3LayerWeights(ctx, layer));
            }
        }
    }

    ~Qwen3Weight() override = default;
};

} // namespace powerserve
