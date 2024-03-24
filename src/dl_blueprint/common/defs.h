#pragma once
//
// Created by dewe on 3/12/24.
//
#include "enum.h"
#include <boost/exception/all.hpp>
#include <boost/exception/exception.hpp>
#include <boost/stacktrace.hpp>


using traced = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;

#define DL_ThrowFromException(e) \
    BOOST_THROW_EXCEPTION(boost::enable_error_info(e) << traced(boost::stacktrace::stacktrace()))
#define DL_ThrowImpl(e, ExceptionClass) DL_ThrowFromException(ExceptionClass(fmt::format("{}", e)))
#define DL_Throw(e) DL_ThrowImpl(e, std::runtime_error)
#define DL_AssertIfTrue(cond, str_) if (not (cond)) DL_Throw(fmt::format("!({})\nMessage:\n{}", #cond, str_))
#define DL_AssertIfTrueF(cond, str_, ...) \
if (not (cond)) DL_Throw(fmt::format("!({})\nMessage:\n{}", #cond, fmt::format(str_, __VA_ARGS__)))

#define DL_AssertIfFalse(cond, str_) if ((cond)) DL_Throw(fmt::format("{}\nMessage:\n{}", #cond, str_))
#define DL_AssertIfFalseF(cond, str_, ...) \
if ((cond)) DL_Throw(fmt::format("{}\nMessage:\n{}", #cond, fmt::format(str_, __VA_ARGS__)))

namespace dlb {

    DL_CREATE_ENUM(WeightParamType, orthogonal, xavier_uniform, xavier_normal, constant)
    struct WeightInit
    {
        WeightParamType type{WeightParamType::orthogonal};
        double gain{std::sqrt(2.f)};
    };
    DL_CREATE_ENUM(ActivationFunction, tanh, relu, leaky_relu, sigmoid)
}