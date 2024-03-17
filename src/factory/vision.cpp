//
// Created by dewe on 3/13/24.
//
#include "vision.h"
#include "helper.h"


namespace dlb {

    template<class T>
    void Make(YAML::Node const &node,
              int64_t in_features,
              T &option) {

        const auto filters = node["filters"].as<std::vector<int64_t>>();
        const auto maxSize = static_cast<int64_t>(filters.size());
        for (auto &&[i, filter]: ranges::view::enumerate(filters)) {
            typename T::TorchOptionType conv2dOptions{in_features, filter,
                                                      GetOptionalValue<int64_t>("kernels",
                                                                                node,
                                                                                static_cast<int64_t>(i),
                                                                                maxSize).value()};

            conv2dOptions.stride(GetValue("strides", node, 1L, static_cast<int64_t>(i), maxSize));
            if (auto padding = GetValueNode("paddings", node, static_cast<int64_t>(i), maxSize)) {
                auto paddingStr = padding->as<std::string>();
                if (std::ranges::all_of(paddingStr, isdigit)) {
                    conv2dOptions.padding(std::stoi(paddingStr));
                } else {
                    if constexpr (std::same_as<typename T::TorchOptionType, torch::nn::Conv2dOptions>) {
                        const std::ranges::input_range auto paddingStrLower =
                                tl::to<std::string>(std::views::transform(paddingStr, tolower));
                        if (paddingStrLower == "same") {
                            conv2dOptions.padding(torch::kSame);
                        } else if (paddingStrLower == "valid") {
                            conv2dOptions.padding(torch::kValid);
                        } else {
                            throw std::runtime_error(fmt::format("Invalid Padding: {}", paddingStrLower));
                        }
                    }
                }
            }

            typename T::ElementOptionType options{std::move(conv2dOptions)};
            DecodeBaseModuleOption(options, node, static_cast<size_t>(i), maxSize);
            in_features = filter;

            option.impl.emplace_back(options);
        }
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              CNNOption &option) {

        Make<>(node, in_features, option);
    }

    void Make(YAML::Node const &node,
              int64_t in_features,
              CNNTransposeOption &option) {
        Make<>(node, in_features, option);
    }
}