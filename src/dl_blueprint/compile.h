#pragma once
//
// Created by dewe on 3/17/24.
//
#include "dl_blueprint/blueprint.h"
#include "dl_blueprint/variable.h"


namespace dlb {
    using FeatureInput = torch::OrderedDict<std::string, Shape>;

    Blueprint Build(const FeatureInput &inputShape,
                           const YAML::Node &config);

    Node CompileNode(std::string const &parentName,
                     const YAML::Node &node,
                     FeatureInput inputShape);

    std::pair<Shape, Forwardable> CompileSequential(std::string const &parentName,
                                                    const YAML::Node &sequence,
                                                    const FeatureInput &inputShape);

    std::pair<Shape, Forwardable> CompileModule(std::string const &parentName,
                                                const YAML::Node &sequence,
                                                const FeatureInput &inputShape);

    std::pair<Shape, torch::nn::AnyModule> ExtractModuleNode(Variable const &variable,
                                                             YAML::Node const &layer,
                                                             FeatureInput &inputShape);
}