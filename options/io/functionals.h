#pragma once
//
// Created by dewe on 3/17/24.
//
#include <cinttypes>


namespace torch::nn {

    class Dim {
    public:
        Dim(int64_t x) : value(x + 1) {}

        int64_t dim() const { return value; }
    private:
        int64_t value;
    };

    struct IndexOption {
        int64_t index{0};
        Dim dim{0};
    };
    using Index = int64_t;
}

namespace dlb {
//    struct Index : OptionsHolder<int64_t> {
//        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
//            return {impl.out_features()};
//        }
//    };

//    struct ConcatDim : OptionsHolder<int64_t> {
//        Shape GetOutputSize(const dlb::Shape &inputShape) const final {
//            return {impl.out_features()};
//        }
//    };

    struct SelectOption : OptionsHolder<torch::nn::IndexOption> {
        SelectOption(torch::nn::IndexOption options) : OptionsHolder<torch::nn::IndexOption>(std::move(options)) {}

        Shape GetOutputSize(const Shape &inputShape) const final {
            auto outShape = inputShape;
            outShape[impl.dim.dim()] = 1;
            return outShape;
        }
    };

}