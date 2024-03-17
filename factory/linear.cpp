//
// Created by dewe on 3/13/24.
//
#include "linear.h"
#include "range/v3/range.hpp"
#include "helper.h"


namespace dlb {
    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<LinearOptions> &options,
              int64_t dim,
              int64_t index = 0,
              int64_t maxSize = -1) {
        torch::nn::LinearOptions linearOptions{in_features, dim};
        linearOptions.bias(GetValue<bool>("new_bias", node, false, index, maxSize));
        options = LinearOptions{linearOptions};
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<LinearOptions> &options) {
        Make(node, in_features, options, node["in_features"].as<int64_t>());
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              FCNNOption &option) {

        const auto dims = node["dims"].as<std::vector<int64_t>>();
        const auto maxSize = static_cast<int64_t>(dims.size());
        for (auto &&[i, dim]: ranges::view::enumerate(dims)) {
            const auto index = static_cast<int64_t>(i);

            std::optional<LinearOptions> options;
            Make(node, in_features, options, dim);

            DecodeBaseModuleOption(*options, node, index, maxSize);
            in_features = dim;

            option.impl.emplace_back(std::move(*options));
        }
    }
}