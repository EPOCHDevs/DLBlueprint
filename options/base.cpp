//
// Created by dewe on 3/12/24.
//
#include "base.h"


namespace dlb {

    void InitializeWeight(WeightInit const& weightInit, torch::Tensor &param) {
        const double gain = weightInit.gain;
        switch (weightInit.type) {
            case WeightParamType::orthogonal:
                torch::nn::init::orthogonal_(param, gain);
                break;
            case WeightParamType::xavier_uniform:
                torch::nn::init::xavier_uniform_(param, gain);
                break;
            case WeightParamType::xavier_normal:
                torch::nn::init::xavier_normal_(param, gain);
                break;
            case WeightParamType::constant:
                torch::nn::init::constant_(param, gain);
                break;
            default:
                break;
        }
    }

    torch::nn::Functional GetModule(ActivationFunction type) {
        switch (type) {
            case ActivationFunction::tanh:
                return torch::nn::Functional{torch::nn::Tanh()};
            case ActivationFunction::relu:
                return torch::nn::Functional{torch::nn::ReLU()};
            case ActivationFunction::leaky_relu:
                return torch::nn::Functional{torch::nn::LeakyReLU()};
            case ActivationFunction::sigmoid:
                return torch::nn::Functional{torch::nn::Sigmoid()};
            default:
                return nullptr;
        }
    }


}