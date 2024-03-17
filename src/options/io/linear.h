#pragma once
//
// Created by dewe on 3/13/24.
//
#include "../base.h"


namespace dlb {
    struct LinearOptions : OptionsHolder<torch::nn::LinearOptions> {

        LinearOptions(torch::nn::LinearOptions args) : OptionsHolder<torch::nn::LinearOptions>(args) {}

        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
            return {impl.out_features()};
        }
    };

    using FCNNOption = OptionsSeqHolder<LinearOptions>;
}