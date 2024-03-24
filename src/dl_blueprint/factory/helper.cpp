//
// Created by dewe on 3/13/24.
//
#include "helper.h"


namespace YAML {
    Node convert<dlb::WeightInit>::encode(dlb::WeightInit const &init) {
        Node node;
        node["type"] = dlb::WeightParamTypeWrapper::ToString(init.type);
        node["gain"] = init.gain;
        return node;
    }

    bool convert<dlb::WeightInit>::decode(YAML::Node const &node, dlb::WeightInit &weight_init) {
        if (auto type = node["type"]) {
            weight_init.type = dlb::WeightParamTypeWrapper::FromString(type.as<std::string>());
        }
        if (auto gain = node["gain"]) {
            weight_init.gain = gain.as<double>();
        }
        return true;
    }
}

namespace dlb {


    std::optional<YAML::Node> GetValueNode(std::string const &key,
                                           YAML::Node const &node,
                                           int64_t i,
                                           int64_t arraySize) {
        if (auto value = node[key]) {
            if (value.IsScalar()) {
                return value;
            } else if (value.IsSequence()) {
                DL_AssertIfTrue(value.size() == arraySize && i < arraySize, "Invalid arguments");
                return value[i];
            }
            DL_ThrowFromException(std::invalid_argument("node must be a scalar or sequence."));
        }
        return {};
    }

    void DecodeBaseModuleOption(BaseModuleOption &option,
                                YAML::Node const &node,
                                int64_t i,
                                int64_t arraySize) {
        if (auto wit = GetValueNode("weight_init", node, i, arraySize)) {
            option.weight_init = wit->as<WeightInit>();
        }
        if (auto bit = GetValueNode("bias_init", node, i, arraySize)) {
            option.bias_init = bit->as<WeightInit>();
        }
        if (auto act = GetValueNode("activations", node, i, arraySize)) {
            option.activations = ActivationFunctionWrapper::FromString(act->as<std::string>());
        }
        if (auto fl = GetValueNode("flatten", node, i, arraySize)) {
            option.flatten = fl->as<bool>();
        }
    }

    torch::nn::Conv2dOptions::padding_t GetPadding(BaseModuleOption &option,
                                                   YAML::Node const &node,
                                                   int64_t i,
                                                   int64_t arraySize) {
        if (auto padding = GetValueNode("paddings", node, i, arraySize)) {
            auto paddingStr = padding->as<std::string>();
            if (std::ranges::all_of(paddingStr, isdigit)) {
                return std::stoi(paddingStr);
            } else {
                const std::ranges::input_range auto paddingStrLower =
                        tl::to<std::string>(std::views::transform(paddingStr, tolower));
                if (paddingStrLower == "same") {
                    return torch::kSame;
                } else if (paddingStrLower == "valid") {
                    return torch::kValid;
                } else {
                    throw std::runtime_error(fmt::format("Invalid Padding: {}", paddingStrLower));
                }
            }
        }
        return 0;
    }
}