#pragma once
//
// Created by dewe on 3/16/24.
//
#include "models/base.h"


namespace dlb {
    struct NodeImpl {
        Forwardable module;
        std::unordered_map<std::string, std::unique_ptr<NodeImpl>> children;
        std::string key;
        std::string input;

        NodeImpl(Forwardable mod,
                 std::string key,
                 std::string input)
                : module(std::move(mod)), key(std::move(key)), input(std::move(input)) {}
    };
    using Node = std::unique_ptr<NodeImpl>;
}