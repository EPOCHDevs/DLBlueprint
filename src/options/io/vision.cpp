//
// Created by dewe on 3/13/24.
//

#include "vision.h"


namespace dlb {

    template<typename OptionType>
    torch::ExpandingArray<2> Conv2dOptionsT<OptionType>::GetPadding(torch::ExpandingArray<2> const &spatialSize) const {
        const auto padding = this->impl.padding();

        if constexpr (std::same_as<OptionType, torch::nn::ConvTranspose2dOptions>) {
            return padding;
        }
        else
        {
            if (std::holds_alternative<torch::enumtype::kValid>(padding)) {
                return {0, 0};
            } else if (std::holds_alternative<torch::enumtype::kSame>(padding)) {
                std::vector<int64_t> paddings{2};
                for (int j = 0; j < 2; ++j) {
                    auto d = this->impl.dilation()->at(j);
                    auto k = this->impl.kernel_size()->at(j);
                    auto s = this->impl.stride()->at(j);
                    auto l = spatialSize->at(j);

                    auto total_padding = l * (s - 1) - s + d * (k - 1) + 1;
                    auto left_pad = static_cast<int>(total_padding / 2);
                    paddings[j] = left_pad;
                }
                return paddings;
            }
            return std::get<torch::ExpandingArray<2>>(padding);
        }
    }

    template<typename OptionType>
    Shape Conv2dOptionsT<OptionType>::GetOutputSize(const dlb::Shape &inputShape) const {
        if (this->flatten) {
            return {-1};
        }

        int64_t height = inputShape[2];
        int64_t width = inputShape[3];

        torch::ExpandingArray<2> paddings = GetPadding({height, width});
        return {
                this->impl.out_channels(),
                GetOutputLength(height, paddings->at(0), 0),
                GetOutputLength(width, paddings->at(1), 1)
        };
    }

    template<typename OptionType>
    Shape CNNOptionT<OptionType>::GetOutputSize(const Shape &inputShape) const {
        Shape outputShape = inputShape;
        for (auto const &layer: this->impl) {
            outputShape = layer.GetOutputSize(outputShape);
        }
        return outputShape;
    }

    template
    class Conv2dOptionsT<torch::nn::Conv2dOptions>;

    template
    class CNNOptionT<Conv2dOptions>;

    template
    class Conv2dOptionsT<torch::nn::ConvTranspose2dOptions>;

    template
    class CNNOptionT<ConvTranspose2dOptions>;
}
