//
// Created by dewe on 3/16/24.
//
#include "sparse.h"
#include "helper.h"


namespace dlb {
    void Make(YAML::Node const &node,
              int64_t in_features,
              EmbeddingsOption &option)
    {
        const auto dims = node["dims"].as<std::vector<int64_t>>();
        const auto maxSize = static_cast<int64_t>(dims.size());
        for (auto && [i, dim]: ranges::view::enumerate(dims))
        {
            const auto index = static_cast<int64_t>(i);

            torch::nn::EmbeddingOptions embeddingOptions{in_features, dim};
            embeddingOptions.padding_idx(GetOptionalValue<int64_t>("padding_idx", node, index, maxSize));
            embeddingOptions.max_norm(GetOptionalValue<int64_t>("max_norm", node, index, maxSize));

            if (auto norm_type = GetValueNode("norm_type", node, index, maxSize))
            {
                embeddingOptions.norm_type(norm_type->as<double>());
            }

            if (auto scale_grad_by_freq = GetValueNode("scale_grad_by_freq", node, index, maxSize))
            {
                embeddingOptions.scale_grad_by_freq(scale_grad_by_freq->as<bool>());
            }

            if (auto sparse = GetValueNode("sparse", node, index, maxSize))
            {
                embeddingOptions.sparse(sparse->as<bool>());
            }

            EmbeddingOptions options{embeddingOptions};
            DecodeBaseModuleOption(options, node, index, maxSize);
            in_features = dim;

            option.impl.emplace_back(options);
        }
    }
}