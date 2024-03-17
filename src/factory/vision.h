#pragma once
//
// Created by dewe on 3/13/24.
//
#include "options/options.h"


namespace dlb {
    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<BasicBlockOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<BottleneckOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<ResNetOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<Conv2dOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<ConvTranspose2dOptions> &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              CNNOption &option);

    void Make(YAML::Node const &node,
              int64_t in_features,
              CNNTransposeOption &option);
}