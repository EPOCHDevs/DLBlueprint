#pragma once
//
// Created by dewe on 3/17/24.
//
#include "models/base.h"
#include "options/options.h"


namespace torch::nn {
    class SelectImpl : public Module {
    public:
        SelectImpl(IndexOption options) : m_option(std::move(options)) {}

        torch::Tensor forward(const torch::Tensor &x)
        {
            return torch::select(x, m_option.dim.dim(), m_option.index);
        }

    private:
        IndexOption m_option;
    };

    struct FirstImpl : SelectImpl {
        FirstImpl(Dim dim) : SelectImpl(IndexOption{0, dim}) {}
    };

    struct LastImpl : SelectImpl {
        LastImpl(Dim dim) : SelectImpl(IndexOption{-1, dim}) {}
    };
}