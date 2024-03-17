#pragma once
//
// Created by dewe on 3/17/24.
//
#include "options/options.h"


namespace dlb {
    void Make(YAML::Node const &node,
              std::optional<torch::nn::Index> &option);

    void Make(YAML::Node const &node,
              std::optional<torch::nn::Dim> &option);

    void Make(YAML::Node const &node,
              std::optional<torch::nn::IndexOption> &option);
}