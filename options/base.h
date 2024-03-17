#pragma once
//
// Created by dewe on 3/12/24.
//
#include "string"
#include "cmath"
#include "yaml-cpp/yaml.h"
#include "optional"
#include "torch/torch.h"
#include "common/defs.h"
#include "yaml-cpp/yaml.h"


namespace dlb {
    void InitializeWeight(WeightInit const&, torch::Tensor &param);

    torch::nn::Functional GetModule(ActivationFunction );

    using Shape = std::vector<int64_t>;

    struct BaseModuleOption {
        std::optional<WeightInit> weight_init{};
        std::optional<WeightInit> bias_init{};
        std::optional<ActivationFunction> activations;
        bool flatten{false};

        virtual void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const {
            ValidateInputDim(2, inputShape);
            ValidateBatchSize(batchSize, inputShape);
        }

        static void ValidateBatchSize(int64_t batchSize, Shape const &inputShape) {
            AssertIfTrue(inputShape.front() == batchSize, "InvalidBatchSize");
        }

        static void ValidateInputDim(int64_t dim, Shape const &inputShape) {
            AssertIfTrue(inputShape.size() == dim, "InvalidInputDim");
        }

        virtual Shape GetOutputSize(Shape const &inputShape) const = 0;
    };

    template<class Module>
    void InitializeWeightBias(Module && sub_module, BaseModuleOption const &option) {
        for (auto &weight: sub_module->named_parameters()) {
            if (weight.key().find("bias") != std::string::npos && option.bias_init)
                InitializeWeight(*option.bias_init, weight.value());
            else if (weight.key().find("weight") != std::string::npos && option.weight_init)
                InitializeWeight(*option.weight_init, weight.value());
        }
    }

    template<class T>
    struct OptionsHolder : BaseModuleOption {
        using TorchOptionType = T;

        T impl;

        OptionsHolder(T args):impl(std::move(args)){}
        // TODO: Serialize torch options to Node
    };

    template<class T>
    struct OptionsSeqHolder : BaseModuleOption {
        using TorchOptionType = typename T::TorchOptionType;
        using ElementOptionType = T;

        std::vector<T> impl;

        auto begin() const { return impl.begin(); }
        auto end() const { return impl.end(); }

        Shape GetOutputSize(const dlb::Shape &inputShape) const override {
            return impl.back().GetOutputSize(inputShape);
        }
    };

}