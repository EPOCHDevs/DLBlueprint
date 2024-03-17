#pragma once
//
// Created by dewe on 3/17/24.
//
#include "models/base.h"
#include "options/options.h"


namespace torch::nn {
    class ReduceImpl : public Module {
    public:
        ReduceImpl(Dim options) : m_dim(std::move(options)) {}

        virtual torch::Tensor reduce(const std::vector<torch::Tensor> &x) = 0;

        int64_t GetDim() const { return m_dim.dim(); }

    private:
        Dim m_dim;
    };

    class ConcatImpl : public ReduceImpl {
    public:
        ConcatImpl(Dim options) : ReduceImpl(std::move(options)) {}

        torch::Tensor reduce(const std::vector<torch::Tensor> &x) final {
            return torch::concat(x, GetDim());
        }
    };

    struct HStackImpl : public ReduceImpl {
        torch::Tensor reduce(const std::vector<torch::Tensor> &x) final {
            return torch::hstack(x);
        }
    };

    struct VStackImpl : public ReduceImpl {
        torch::Tensor reduce(const std::vector<torch::Tensor> &x) final {
            return torch::vstack(x);
        }
    };
}