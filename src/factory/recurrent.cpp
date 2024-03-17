//
// Created by dewe on 3/16/24.
//

#include "recurrent.h"
#include "helper.h"


namespace dlb {
    template<class TemplateOptionType>
    TemplateOptionType Make(YAML::Node const &node, int64_t in_features) {
        TemplateOptionType torchOption(in_features,
                                       node["hidden_size"].as<int64_t>());
        SET_OPTIONAL_PARAM(num_layers);
        SET_OPTIONAL_PARAM(dropout);
        SET_OPTIONAL_PARAM(bidirectional);
        SET_OPTIONAL_PARAM(bias);

        if constexpr (std::same_as<TemplateOptionType, torch::nn::RNNOptions>) {
            if (auto nonlinearity = node["nonlinearity"]) {
                auto nonLinearityStr = nonlinearity.as<std::string>();
                std::ranges::transform(nonLinearityStr,
                                       nonLinearityStr.begin(),
                                       tolower);
                if (nonLinearityStr == "tanh") {
                    torchOption.nonlinearity(torch::kTanh);
                } else if (nonLinearityStr == "relu") {
                    torchOption.nonlinearity(torch::kReLU);
                } else {
                    throw std::invalid_argument("Invalid non_linearity: " + nonLinearityStr);
                }
            }
        }

        if constexpr (std::same_as<TemplateOptionType, torch::nn::LSTMOptions>) {
            SET_OPTIONAL_PARAM(proj_size);
        }

        return torchOption;
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<RNNOptions> &option) {

        option = Make<torch::nn::RNNOptions>(node, in_features);
        option->return_all_seq = node["return_all_seq"].as<bool>(false);
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<GRUOptions> &option) {
        option = Make<torch::nn::GRUOptions>(node, in_features);
        option->return_all_seq = node["return_all_seq"].as<bool>(false);
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              std::optional<LSTMOptions> &option) {
        option = Make<torch::nn::LSTMOptions>(node, in_features);
        option->return_all_seq = node["return_all_seq"].as<bool>(false);
    }
}