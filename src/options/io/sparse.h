#pragma once
//
// Created by dewe on 3/13/24.
//
#include "../base.h"


namespace dlb {
    struct EmbeddingOptions : OptionsHolder<torch::nn::EmbeddingOptions> {
        EmbeddingOptions(const torch::nn::EmbeddingOptions &args) : OptionsHolder<torch::nn::EmbeddingOptions>(args) {}

        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
            return {impl.embedding_dim()};
        }
    };

    using EmbeddingsOption = OptionsSeqHolder<EmbeddingOptions>;
}