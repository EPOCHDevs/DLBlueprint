#pragma once
//
// Created by dewe on 3/13/24.
//
#include "yaml-cpp/yaml.h"
#include "optional"
#include "dl_blueprint/common/defs.h"
#include "dl_blueprint/options/base.h"
#include "tl/to.hpp"


namespace dlb {

#define SET_OPTIONAL_PARAM(param_name) \
    torchOption = torchOption. param_name(node[#param_name].as<std::decay_t<decltype(torchOption. param_name())>>(torchOption. param_name()))


    std::optional<YAML::Node>
    GetValueNode(std::string const &key, YAML::Node const &node, int64_t i = 0, int64_t arraySize = -1);

    template<class T>
    std::optional<T> GetOptionalValue(std::string const &key, YAML::Node const &node, int64_t i = 0, int64_t arraySize = -1) {
        if (auto value = GetValueNode(key, node, i, arraySize)) {
            return value->as<T>();
        }
        return std::nullopt;
    }

    template<class T>
    T GetValue(std::string const &key,
               YAML::Node const &node,
               T const &defaultValue,
               int64_t i = 0, int64_t arraySize = -1) {
        return GetOptionalValue<T>(key, node, i, arraySize).value_or(defaultValue);
    }

    void DecodeBaseModuleOption(BaseModuleOption &option,
                                YAML::Node const &node,
                                int64_t i = 0,
                                int64_t arraySize = -1);

    torch::nn::Conv2dOptions::padding_t GetPadding(BaseModuleOption &option,
                                                   YAML::Node const &node,
                                                   int64_t i = 0,
                                                   int64_t arraySize = -1);
}

namespace YAML
{
    template<>
    struct convert<dlb::WeightInit>
    {
        static YAML::Node encode(dlb::WeightInit const&);
        static bool decode(YAML::Node const&, dlb::WeightInit&);
    };
}