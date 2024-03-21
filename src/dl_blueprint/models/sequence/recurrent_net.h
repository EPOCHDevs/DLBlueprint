#pragma once
//
// Created by dewe on 3/13/24.
//
#include "dl_blueprint/models/base.h"
#include "dl_blueprint/options/options.h"


namespace dlb {
    // TODO: Create 2 specializations for batch_first and non batch_first
    template<class TorchModuleT, class OptionTypeT, const char *class_name>
    class RecurrentNetImpl : public ForwardableImpl {
    public:
        constexpr static bool isLSTM = std::same_as<TorchModuleT, torch::nn::LSTM>;
        using HiddenStateType = std::conditional_t<isLSTM, Tensor2, torch::Tensor>;
        using TorchModule = TorchModuleT;
        using OptionType = OptionTypeT;

    public:
        RecurrentNetImpl(OptionType options
        ) : m_model(register_module(class_name, TorchModule{options.impl.batch_first(true)})),
            m_options(std::move(options)) {}

        bool has_state() final {
            return true;
        }

        const BaseModuleOption *GetOption() const final {
            return &m_options;
        }

        std::string GetName() const final {
            return class_name;
        }

        void reset_state(int64_t batchSize) final {
            auto options = m_model->options;
            torch::TensorOptions weightOptions = m_model->all_weights().front().options();
            initialize_hidden_states(options.num_layers() * (options.bidirectional() ? 2 : 1), batchSize,
                                     options.hidden_size(),
                                     weightOptions.requires_grad(std::nullopt));
        }

        torch::Tensor forward(const torch::Tensor &x) override {
            torch::Tensor result;
            repackage_hidden();
            std::tie(result, hx) = m_model->forward(x, hx);
            return m_options.return_all_seq ? result : result.select(1, -1);
        }

    protected:
        TorchModule m_model;
        HiddenStateType hx;
        OptionType m_options;

        void repackage_hidden() {
            if constexpr (isLSTM) {
                hx = {
                        std::get<0>(hx).detach(), std::get<1>(hx).detach()
                };
            } else {
                hx = hx.detach();
            }
        }

        void initialize_hidden_states(int64_t numLayers, int64_t batchSize, int64_t hiddenSize,
                                      torch::TensorOptions const &options) {
            if constexpr (isLSTM) {
                hx = Tensor2{register_buffer("h0", torch::zeros({numLayers, batchSize, hiddenSize}, options)),
                             register_buffer("c0", torch::zeros({numLayers, batchSize, hiddenSize}, options))};
            } else {
                hx = register_buffer("h0", torch::zeros({numLayers, batchSize, hiddenSize}, options));
            }
        }
    };
}