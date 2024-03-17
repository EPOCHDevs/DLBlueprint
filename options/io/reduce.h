#pragma once
//
// Created by dewe on 3/17/24.
//
#include "functionals.h"


namespace dlb {
    using Shapes = std::vector<Shape>;

    struct ReduceOption {
        virtual Shape GetOutputSize(const Shapes &inputShapes) const {
            return std::accumulate(inputShapes.begin() + 1,
                                   inputShapes.end(),
                                   inputShapes[0],
                                   [i = dim.dim()](Shape &&result, Shape const &arg) {
                                       result[i] += arg.at(i);
                                       return result;
                                   });
        }

        torch::nn::Dim dim;
    };
}