//
// Created by dewe on 3/17/24.
//
#include "variable.h"
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>


namespace dlb {
    Variable::Variable(std::string const &prototype) {
        size_t colonPos = prototype.find(':');
        size_t openParensPos = prototype.find('(');
        size_t closeParensPos = prototype.find(')');

        if (colonPos == std::string::npos ||
            (openParensPos != std::string::npos && closeParensPos == std::string::npos) ||
            (openParensPos == std::string::npos && closeParensPos != std::string::npos)) {
            throw std::invalid_argument("Given string has an invalid syntax");
        }

        name = prototype.substr(0, colonPos);
        if (openParensPos != std::string::npos) {
            type = prototype.substr(colonPos + 1, openParensPos - colonPos - 1);
            std::string inputsStr = prototype.substr(openParensPos + 1, closeParensPos - openParensPos - 1);

            typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
            boost::char_separator<char> sep(",");
            tokenizer tokens(inputsStr, sep);
            for (const auto &t: tokens) {
                std::string s = boost::trim_copy(t);
                inputs.emplace_back(s);
            }
        } else {
            type = prototype.substr(colonPos + 1);
        }
    }
}