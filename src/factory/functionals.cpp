//
// Created by dewe on 3/17/24.
//

#include "functionals.h"


namespace dlb {
    void Make(YAML::Node const &node,
              std::optional<torch::nn::Index> &option) {
        option = torch::nn::Index{node["index"].as<int64_t>()};
    }

    void Make(YAML::Node const &node,
              std::optional<torch::nn::Dim> &option) {
        option = torch::nn::Dim{node["dim"].as<int64_t>(0)};
    }

    void Make(YAML::Node const &node,
              std::optional<torch::nn::IndexOption> &option) {
        option = torch::nn::IndexOption{.index=node["index"].as<int64_t>(),
                .dim=node["dim"].as<int64_t>(1)};
    }
}