#pragma once
//
// Created by dewe on 3/13/24.
//
#include "../base.h"
#include "dl_blueprint/models/vision/options.h"


namespace dlb {

    template<typename OptionType>
    struct Conv2dOptionsT : OptionsHolder<OptionType> {

        Conv2dOptionsT(OptionType opts):OptionsHolder<OptionType>(std::move(opts)){}

        void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const final {
            this->ValidateInputDim(4, inputShape);
            this->ValidateBatchSize(batchSize, inputShape);
            AssertIfTrue(inputShape[1] == this->impl.in_channels(), "InvalidChannelSize");
        }

        Shape GetOutputSize(const dlb::Shape &inputShape) const final;

        torch::ExpandingArray<2> GetPadding(torch::ExpandingArray<2> const &spatialSize) const;

        int64_t GetOutputLength(int size, int64_t padding, int64_t index) const {
            const int64_t stride = this->impl.stride()->at(index);
            const int64_t kernel = this->impl.kernel_size()->at(index);
            const int64_t dilation = this->impl.dilation()->at(index);
            return ((size + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
        }
    };

    template<typename OptionType>
    struct CNNOptionT : OptionsSeqHolder<OptionType> {
        void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const final {
            this->impl.front().ValidateInputShape(batchSize, inputShape);
        }

        Shape GetOutputSize(const dlb::Shape &inputShape) const final;
    };

    using Conv2dOptions = Conv2dOptionsT<torch::nn::Conv2dOptions>;
    using CNNOption = CNNOptionT<Conv2dOptions>;

    using ConvTranspose2dOptions = Conv2dOptionsT<torch::nn::ConvTranspose2dOptions>;
    using CNNTransposeOption = CNNOptionT<ConvTranspose2dOptions>;

    extern template class Conv2dOptionsT<torch::nn::Conv2dOptions>;
    extern template class CNNOptionT<Conv2dOptions>;

    extern template class Conv2dOptionsT<torch::nn::ConvTranspose2dOptions>;
    extern template class CNNOptionT<ConvTranspose2dOptions>;

    struct BasicBlockOptions : OptionsHolder<torch::nn::ResNetBlockOptions> {
        void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const final {
            throw std::runtime_error("Not Implemented");
        }

        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
            throw std::runtime_error("Not Implemented");
        }
    };

    struct BottleneckOptions : OptionsHolder<torch::nn::ResNetBlockOptions> {
        void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const final {
            throw std::runtime_error("Not Implemented");
        }

        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
            throw std::runtime_error("Not Implemented");
        }
    };


    struct ResNetOptions : OptionsHolder<torch::nn::ResNetOptions> {
        void ValidateInputShape(int64_t batchSize, Shape const &inputShape) const final {
            dlb::ResNetOptions::ValidateInputDim(4, inputShape);
            dlb::ResNetOptions::ValidateBatchSize(batchSize, inputShape);
            AssertIfTrue(inputShape[1] == 3, "InvalidChannelSize");
        }

        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
            return {
                    inputShape[0], impl.num_classes
            };
        }
    };
}