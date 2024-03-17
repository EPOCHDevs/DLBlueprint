//
// Created by dewe on 3/17/24.
//

#include "compile.h"
#include "blueprint.h"


namespace dlb {

    Blueprint Build(const FeatureInput &inputShape,
               const YAML::Node &config) {
        AssertIfTrue(config.size() == 1 && config.IsMap(), "only one root is permitted");
        return {CompileNode(config.begin()->first.as<std::string>(), config.begin()->second, inputShape)};
    }

    Node CompileNode(std::string const &moduleName,
                     const YAML::Node &node,
                     FeatureInput inputShape) {
        auto seqNode = node["<modules>"];
        AssertIfTrueF(seqNode, "<modules> is a mandatory argument for any node: {}", moduleName);

        auto [shape, module] = CompileModule(moduleName, seqNode, inputShape);
        auto newNode = std::make_unique<NodeImpl>(module, moduleName, inputShape.back().key());
        inputShape.insert(moduleName, shape);

        if (node["<children>"]) {
            for (auto const &child: node["<children>"]) {
                std::string childName = child.first.as<std::string>();
                newNode->children[childName] = CompileNode(childName, child.second, inputShape);
            }
        }

        return newNode;
    }

    std::pair<Shape, torch::nn::AnyModule> ExtractModuleNode(Variable const &variable,
                                                             YAML::Node const &layer,
                                                             const Shape &inputShape) {
        // for 1D feature: [0] is length
        // for 2D feature: [0] is channel_dim
        auto module = MakeModule(variable.type)(layer, inputShape[0]);
        return {module->GetOption()->GetOutputSize(inputShape), torch::nn::AnyModule(module)};
    }

    std::pair<Shape, Forwardable> CompileSequential(std::string const &moduleName,
                                                    const YAML::Node &sequence,
                                                    const FeatureInput &inputShape) {
        torch::nn::Sequential model;
        auto inputShape_ = inputShape.back().value();

        for (auto const &layer: sequence) {
            const auto key = layer.first.as<std::string>();
            const Variable variable(key);

            torch::nn::AnyModule module;
            std::tie(inputShape_, module) = ExtractModuleNode(variable, layer.second, inputShape_);
            model->push_back(variable.name, module);
        }
        std::shared_ptr<ForwardableImpl> module = std::make_shared<BaseSequentialModuleImpl>(moduleName, model);
        return {inputShape_, module};
    }

    std::pair<Shape, Forwardable> CompileModule(std::string const &moduleName,
                                                const YAML::Node &sequenceOrModule,
                                                const FeatureInput &inputShape) {
        if (sequenceOrModule.size() == 1) {
            const auto key = sequenceOrModule.begin()->first.as<std::string>();
            const Variable variable(key);
            auto [shape, module] = ExtractModuleNode(variable, sequenceOrModule.begin()->second,
                                                     inputShape.back().value());
            return {shape, module.ptr<ForwardableImpl>()};
        } else {
            return CompileSequential(moduleName, sequenceOrModule, inputShape);
        }
    }
}