#pragma once
//
// Created by dewe on 3/13/24.
//
#include "../base.h"


namespace dlb {
    template<class TorchRecurrentOptions>
    struct RecurrentOptionsT : OptionsHolder<TorchRecurrentOptions> {
        RecurrentOptionsT(TorchRecurrentOptions opt) :
                OptionsHolder<TorchRecurrentOptions>(std::move(opt)) {}

        void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const final {
            this->ValidateInputDim(3, inputShape);
            // currently assuming only batch_first
            this->ValidateBatchSize(batchSize, inputShape);
            DL_AssertIfTrue(inputShape[2] == this->impl.input_size(), "InvalidChannelSize");
        }

        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
            return return_all_seq ? Shape{inputShape[0], this->impl.hidden_size()} : Shape{
                    this->impl.hidden_size()};
        }

        bool return_all_seq{true};
    };

    using RNNOptions = RecurrentOptionsT<torch::nn::RNNOptions>;
    using GRUOptions = RecurrentOptionsT<torch::nn::GRUOptions>;
    using LSTMOptions = RecurrentOptionsT<torch::nn::LSTMOptions>;
}