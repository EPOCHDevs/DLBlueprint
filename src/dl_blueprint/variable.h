#pragma once
//
// Created by dewe on 3/17/24.
//
#include "string"
#include "vector"

namespace dlb
{
    struct Variable {
        std::string type;
        std::vector<std::string> inputs;
        std::string name;

        Variable(std::string const& prototype);
    };

}