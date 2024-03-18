#pragma once
//
// Created by dewe on 3/17/24.
//
#include "node.h"


namespace dlb {
    using TensorDict = torch::OrderedDict<std::string, torch::Tensor>;

    class Blueprint : public torch::nn::Module {

    public:
        using Visitor = std::function<void(NodeImpl* )>;

        Blueprint(Node node) : m_root(std::move(node)) {
            BFSVisit([this](NodeImpl *node) {
                register_module(node->key, node->module);
            });
        }

        void reset_state(int64_t batchsize) {
            BFSVisit([&](NodeImpl *node) {
                if (node->module->has_state()) {
                    node->module->reset_state(batchsize);
                }
            });
        }

        void to(torch::Device device, bool non_blocking = false) final {
            BFSVisit([&](NodeImpl *node) {
                node->module->to(device, non_blocking);
            });
        }

        void forward(TensorDict& );

    private:
        Node m_root;

        void BFSVisit(Visitor const& visitor);
    };
}