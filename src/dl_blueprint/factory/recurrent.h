#pragma once
//
// Created by dewe on 3/13/24.
//
#include "dl_blueprint/options/options.h"


namespace dlb {
    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<RNNOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<GRUOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<LSTMOptions> &option);
}