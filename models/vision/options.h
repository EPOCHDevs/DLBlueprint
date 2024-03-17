#pragma once
//
// Created by dewe on 3/12/24.
//
#include "torch/torch.h"


namespace torch::nn {
    inline auto BatchNormCreator = [](int64_t arg) -> BatchNorm2d {
        return BatchNorm2d{arg};
    };

    using BatchNormCreatorType = BatchNorm2d (*)(int64_t);

    struct ResNetBlockOptions {
        int64_t inplanes{};
        int64_t planes{};
        int64_t stride{1};
        Sequential downsample;
        int64_t groups{1};
        int64_t base_width{64};
        int64_t dilation{1};
        BatchNormCreatorType norm_layer{BatchNormCreator};

        int64_t Width() const {
            return int(planes * int(base_width / 64.0)) * groups;
        }
    };

    struct ResNetOptions {
        std::vector<int64_t> layers;
        int64_t num_classes = 1000;
        bool zero_init_residual = false;
        int64_t groups = 1;
        int64_t width_per_group = 64;
        std::vector<bool> replace_stride_with_dilation{};
        BatchNormCreatorType norm_layer{BatchNormCreator};

        ResNetOptions &layers_(std::vector<int64_t> const &v) {
            if (not layers.empty()) {
                std::cerr << "overriding ResNetOptions.layers";
            }
            layers = v;
            return *this;
        }

        ResNetOptions &groups_(int64_t v) {
            groups = v;
            return *this;
        }

        ResNetOptions &width_per_group_(int64_t v) {
            width_per_group = v;
            return *this;
        }
    };
}