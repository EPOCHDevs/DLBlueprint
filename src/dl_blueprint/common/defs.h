#pragma once
//
// Created by dewe on 3/12/24.
//
#include "enum.h"
#include <boost/exception/all.hpp>
#include <boost/exception/exception.hpp>
#include <boost/stacktrace.hpp>


using traced = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;

#define ThrowFromException(e) \
    BOOST_THROW_EXCEPTION(boost::enable_error_info(e) << traced(boost::stacktrace::stacktrace()))
#define ThrowImpl(e, ExceptionClass) ThrowFromException(ExceptionClass(fmt::format("{}", e)))
#define Throw(e) ThrowImpl(e, std::runtime_error)
#define AssertIfTrue(cond, str_) if (not (cond)) Throw(fmt::format("!({})\nMessage:\n{}", #cond, str_))
#define AssertIfTrueF(cond, str_, ...) \
if (not (cond)) Throw(fmt::format("!({})\nMessage:\n{}", #cond, fmt::format(str_, __VA_ARGS__)))

#define AssertIfFalse(cond, str_) if ((cond)) Throw(fmt::format("{}\nMessage:\n{}", #cond, str_))
#define AssertIfFalseF(cond, str_, ...) \
if ((cond)) Throw(fmt::format("{}\nMessage:\n{}", #cond, fmt::format(str_, __VA_ARGS__)))

namespace dlb {

    CREATE_ENUM(WeightParamType, orthogonal, xavier_uniform, xavier_normal, constant)
    struct WeightInit
    {
        WeightParamType type{WeightParamType::orthogonal};
        double gain{std::sqrt(2.f)};
    };
    CREATE_ENUM(ActivationFunction, tanh, relu, leaky_relu, sigmoid)
}