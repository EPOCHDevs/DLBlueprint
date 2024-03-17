#pragma once
//
// Created by dewe on 3/12/24.
//
#include "options.h"


namespace torch::nn
{
    inline torch::nn::Conv2d
    conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride = 1, int64_t groups = 1, int64_t dilation = 1) {
        return torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_planes, out_planes, 3)
                        .stride(stride)
                        .padding(dilation)
                        .groups(groups)
                        .bias(false)
                        .dilation(dilation)
        );
    }

    inline torch::nn::Conv2d conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride = 1) {
        return torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_planes, out_planes, 1)
                        .stride(stride)
                        .bias(false)
        );
    }

    class BasicBlockImpl : public torch::nn::Module {
    public:
        constexpr static int expansion = 1;

        BasicBlockImpl(const ResNetBlockOptions &options);

        torch::Tensor forward(torch::Tensor x);

        nn::BatchNorm2d& _bn2()
        {
            return bn2;
        }

    private:
        Conv2d conv1{nullptr}, conv2{nullptr};
        BatchNorm2d bn1{nullptr}, bn2{nullptr};
        Sequential downsample{nullptr};
        ReLU relu{nullptr};
        int64_t stride;
    };
    TORCH_MODULE(BasicBlock);

    class BottleneckImpl : public torch::nn::Module {
    public:
        constexpr static int expansion = 4;

        BottleneckImpl(const ResNetBlockOptions &options);

        torch::Tensor forward(torch::Tensor x);

        nn::BatchNorm2d& _bn3()
        {
            return bn3;
        }

    private:
        nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        Sequential downsample{nullptr};
        nn::ReLU relu{nullptr};
        int64_t stride;
    };
    TORCH_MODULE(Bottleneck);

    template<class BlockType >
    class ResNetImpl : public nn::Module {
    public:
        ResNetImpl(ResNetOptions const&);

        Tensor forward(Tensor x);

    private:
        nn::Conv2d conv1{nullptr};
        nn::BatchNorm2d bn1{nullptr};
        nn::ReLU relu{nullptr};
        nn::MaxPool2d maxpool{nullptr};
        nn::Sequential layer1{nullptr};
        nn::Sequential layer2{nullptr};
        nn::Sequential layer3{nullptr};
        nn::Sequential layer4{nullptr};
        nn::AdaptiveAvgPool2d avgpool{nullptr};
        nn::Linear fc{nullptr};
        nn::Sequential block;
        int64_t inplanes;
        int64_t dilation;
        int64_t groups;
        int64_t base_width;
        BatchNormCreatorType _norm_layer;

        Sequential MakeLayer(int64_t planes, int64_t blocks, int64_t stride=1, bool dilate=false);
    };

    class Resnet18Impl : public ResNetImpl<BasicBlock>
    {
        Resnet18Impl(ResNetOptions options): ResNetImpl<BasicBlock>(options.layers_({2, 2, 2, 2})){}
    };
    TORCH_MODULE(Resnet18);

    class Resnet34Impl : public ResNetImpl<BasicBlock>
    {
        Resnet34Impl(ResNetOptions options): ResNetImpl<BasicBlock>(options.layers_({3, 4, 6, 3})){}
    };
    TORCH_MODULE(Resnet34);

    class Resnet50Impl : public ResNetImpl<Bottleneck>
    {
        Resnet50Impl(ResNetOptions options): ResNetImpl<Bottleneck>(options.layers_({3, 4, 6, 3})){}
    };
    TORCH_MODULE(Resnet50);

    class Resnet101Impl : public ResNetImpl<Bottleneck>
    {
        Resnet101Impl(ResNetOptions options): ResNetImpl<Bottleneck>(options.layers_({3, 4, 23, 3})){}
    };
    TORCH_MODULE(Resnet101);

    class Resnet152Impl : public ResNetImpl<Bottleneck>
    {
        Resnet152Impl(ResNetOptions options): ResNetImpl<Bottleneck>(options.layers_({3, 8, 36, 3})){}
    };
    TORCH_MODULE(Resnet152);

    class Resnext50_32x4dImpl : public ResNetImpl<Bottleneck> {
        Resnext50_32x4dImpl(ResNetOptions options) : ResNetImpl<Bottleneck>(
                options.layers_({3, 4, 6, 3}).groups_(32).width_per_group_(4)) {}
    };
    TORCH_MODULE(Resnext50_32x4d);

    class Resnext101_32x8dImpl : public ResNetImpl<Bottleneck> {
        Resnext101_32x8dImpl(ResNetOptions options) : ResNetImpl<Bottleneck>(
                options.layers_({3, 4, 23, 3}).groups_(32).width_per_group_(8)) {}
    };
    TORCH_MODULE(Resnext101_32x8d);

    class WideResnet50_2Impl : public ResNetImpl<Bottleneck> {
        WideResnet50_2Impl(ResNetOptions options) : ResNetImpl<Bottleneck>(
                options.layers_({3, 4, 6, 3}).width_per_group_(64 * 2)) {}
    };
    TORCH_MODULE(WideResnet50_2);

    class Wide_ResNet101_2Impl : public ResNetImpl<Bottleneck> {
        Wide_ResNet101_2Impl(ResNetOptions options) : ResNetImpl<Bottleneck>(
                options.layers_({3, 4, 23, 3}).width_per_group_(64 * 2)) {}
    };
    TORCH_MODULE(Wide_ResNet101_2);

    extern template class ResNetImpl<BasicBlock>;
    extern template class ResNetImpl<Bottleneck>;

    using BasicBlockResnetImpl = ResNetImpl<BasicBlock>;
    using BottleneckResnetImpl = ResNetImpl<Bottleneck>;

    TORCH_MODULE(BasicBlockResnet);
    TORCH_MODULE(BottleneckResnet);
}