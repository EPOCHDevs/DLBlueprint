//
// Created by dewe on 3/12/24.
//
#include "resnet.h"
#include "common/defs.h"


namespace torch::nn {
    BasicBlockImpl::BasicBlockImpl(const ResNetBlockOptions &options)
            : conv1(register_module("conv1", conv3x3(options.inplanes, options.planes, options.stride))),
              bn1(register_module("bn1", options.norm_layer(options.planes))),
              relu(register_module("relu", nn::ReLU(nn::ReLUOptions().inplace(true)))),
              conv2(register_module("conv2", conv3x3(options.planes, options.planes))),
              bn2(register_module("bn2", options.norm_layer(options.planes))),
              stride(options.stride) {
        AssertIfFalse(options.groups != 1 or options.base_width != 64,
                      "ValueError: BasicBlock only supports groups=1 and base_width=64");
        AssertIfFalse(options.dilation > 1,
                      "NotImplementedError: Dilation > 1 not supported in BasicBlock");

        if (options.downsample) {
            downsample = register_module("downsample", options.downsample);
        }
    }


    Tensor BasicBlockImpl::forward(Tensor x) {
        Tensor identity = x;

        x = relu(bn1(conv1(x)));
        x = bn2(conv2(x));

        if (downsample) {
            identity = downsample->forward(x);
        }

        x += identity;
        x = relu(x);

        return x;
    }

    BottleneckImpl::BottleneckImpl(const ResNetBlockOptions &options)
            :
            conv1(register_module("conv1", conv1x1(options.inplanes, options.Width()))),
            bn1(register_module("bn1", options.norm_layer(options.Width()))),
            conv2(register_module("conv2", conv3x3(options.Width(),
                                                   options.Width(),
                                                   options.stride,
                                                   options.groups,
                                                   options.dilation))),
            bn2(register_module("bn2", options.norm_layer(options.Width()))),
            conv3(register_module("conv3",
                                  conv1x1(options.Width(), options.planes * expansion))),
            bn3(register_module("bn3", options.norm_layer(options.planes * expansion))),
            relu(register_module("relu", nn::ReLU(nn::ReLUOptions().inplace(true)))),
            downsample(register_module("downsample", options.downsample)),
            stride(options.stride) {}

    Tensor BottleneckImpl::forward(Tensor x) {
        Tensor identity = x;

        x = relu(bn1(conv1(x)));
        x = relu(bn2(conv2(x)));
        x = bn3(conv3(x));

        if (downsample) {
            identity = downsample->forward(x);
        }

        x += identity;
        x = relu(x);

        return x;
    }

    template<class BlockType>
    ResNetImpl<BlockType>::ResNetImpl(const ResNetOptions &options)
            : inplanes(64), dilation(1), groups(options.groups), base_width(options.width_per_group),
            _norm_layer(options.norm_layer){
        conv1 = register_module("conv1",
                                nn::Conv2d(nn::Conv2dOptions(3, inplanes, 7).stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", options.norm_layer(inplanes));
        relu = register_module("relu", nn::ReLU(nn::ReLUOptions().inplace(true)));
        maxpool = register_module("maxpool", nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2).padding(1)));

        layer1 = register_module("layer1", MakeLayer(64, options.layers[0]));
        layer2 = register_module("layer2", MakeLayer(128, options.layers[1], 2,
                                                     options.replace_stride_with_dilation[0]));
        layer3 = register_module("layer3", MakeLayer(256, options.layers[2], 2,
                                                     options.replace_stride_with_dilation[1]));
        layer4 = register_module("layer4", MakeLayer(512, options.layers[3], 2,
                                                     options.replace_stride_with_dilation[2]));

        avgpool = register_module("avgpool", nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc = register_module("fc", nn::Linear(512 * BlockType::Impl::expansion, options.num_classes));

        auto init_bn = [](auto *m) {
            m->weight = nn::init::constant_(m->weight, 1);
            m->bias = nn::init::constant_(m->bias, 0);
        };

        // Initialize weights
        for (auto &m: modules()) {
            if (auto *conv = dynamic_cast<nn::Conv2dImpl *>(m.get())) {
                conv->weight = nn::init::kaiming_normal_(conv->weight, 0, torch::kFanOut, torch::kReLU);
            } else if (auto *bn = dynamic_cast<nn::BatchNorm2dImpl *>(m.get())) {
                init_bn(bn);
            } else if (auto *group_bn = dynamic_cast<nn::BatchNorm2dImpl *>(m.get())) {
                init_bn(group_bn);
            }
        }

        auto init_block_bn = [](auto& m) {
            if (m->weight.defined()) {
                nn::init::constant_(m->weight, 0);
            }
        };

        // Zero-initialize the last BN in each residual branch
        if (options.zero_init_residual) {
            for (auto &m: modules()) {
                if (auto *bottleneck = dynamic_cast<nn::BottleneckImpl *>(m.get())) {
                    init_block_bn(bottleneck->_bn3());
                } else if (auto *basicblock = dynamic_cast<nn::BasicBlockImpl *>(m.get())) {
                    init_block_bn(basicblock->_bn2());
                }
            }
        }
    }

    template<class BlockType>
    Tensor ResNetImpl<BlockType>::forward(Tensor x) {
        x = conv1(x);
        x = bn1(x);
        x = relu(x);
        x = maxpool(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = avgpool(x);
        x = torch::flatten(x, 1);
        x = fc(x);

        return x;
    }

    template<class BlockType>
    Sequential ResNetImpl<BlockType>::MakeLayer(int64_t planes, int64_t blocks, int64_t stride, bool dilate) {
        auto norm_layer = _norm_layer;
        nn::Sequential downsample{nullptr};
        int64_t previous_dilation = dilation;
        if (dilate) {
            dilation *= stride;
            stride = 1;
        }
        if (stride != 1 || inplanes != (planes * BlockType::Impl::expansion)) {
            downsample = nn::Sequential(
                    conv1x1(inplanes, planes * BlockType::Impl::expansion, stride),
                    norm_layer(planes * BlockType::Impl::expansion)
            );
        }

        Sequential layers{};
        layers->push_back(
                BlockType{ResNetBlockOptions{.inplanes=inplanes,
                                       .planes=planes,
                                       .stride=stride,
                                       .downsample=downsample,
                                       .groups=groups,
                                       .base_width=base_width,
                                       .dilation=previous_dilation,
                                       .norm_layer=_norm_layer}});

        inplanes = planes * BlockType::Impl::expansion;
        for (int64_t _: std::views::iota(1L, blocks)) {
            layers->push_back(
                    BlockType(ResNetBlockOptions{.inplanes=inplanes,
                            .planes=planes,
                            .groups=groups,
                            .base_width=base_width,
                            .dilation=dilation,
                            .norm_layer=_norm_layer})
            );
        }
        return layers;
    }

    template
    class ResNetImpl<BasicBlock>;

    template
    class ResNetImpl<Bottleneck>;
} // namespace torch::nn
