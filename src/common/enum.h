#pragma once
#include "fmt/format.h"
#include <map>
#include <string>
#include <vector>
#include "ranges"
#include "boost/algorithm/string.hpp"
#include <range/v3/all.hpp>


template<class EnumClass> inline auto Parse(const char* str)
{
    std::map<EnumClass, std::string> stringEnum;
    std::map<std::string, EnumClass> enumAsString;

    auto strView = std::string_view(str);
    auto splitRange = [](auto rng)
    {
        return std::string_view(rng.begin(), rng.end()); // C++20 (subrange::data(), size())
    };

    int8_t numValue = 0;
    for (auto const& rng : strView | ranges::views::split(','))
    {
        std::string_view elementView(&*rng.begin(), ranges::distance(rng));
        std::string elementClone;

        auto value_pos = elementView.find('=');
        if (value_pos == std::string::npos)
        {
            elementClone = elementView;
        }
        else
        {
            elementClone = elementView.substr(0, value_pos);
            std::string valueClone{ elementView.substr(value_pos + 1) };
            boost::trim(valueClone);
            numValue = std::stoi(valueClone);
        }
        boost::trim(elementClone);
        stringEnum[static_cast<EnumClass>(numValue)] = elementClone;
        enumAsString[elementClone] = static_cast<EnumClass>(numValue);
        ++numValue;
    }
    stringEnum[EnumClass::Length] = "Null";
    enumAsString["Null"] = EnumClass::Length;
    return std::pair{ stringEnum, enumAsString };
}


#define CREATE_ENUM_COMMON(EnumClass, NumType, ...) \
    enum class EnumClass : NumType { __VA_ARGS__, Length }; \
    \
    struct EnumClass##Wrapper { \
        EnumClass##Wrapper() { Initialize(); } \
        static inline std::string ToString(EnumClass enumClass) { return stringEnum.at( enumClass); } \
        static inline EnumClass FromString(std::string const& enumClassAsString) { \
            auto enumAsStringIt = enumAsString.find(enumClassAsString); \
            AssertIfFalse(enumAsStringIt == enumAsString.end(), fmt::format("{} is not a valid enum for {}", enumClassAsString, #EnumClass)); \
            return enumAsStringIt->second; \
        } \
        static inline bool Is##EnumClass(std::string const& enumClassAsString) { return enumAsString.contains(enumClassAsString); } \
    private: \
        inline static std::map<EnumClass, std::string> stringEnum; \
        inline static std::map<std::string, EnumClass> enumAsString; \
        void Initialize() const { std::tie(stringEnum, enumAsString) = Parse<EnumClass>(#__VA_ARGS__);  } \
    }; \
    inline std::ostream& operator<<(std::ostream& os, EnumClass enumClass) { \
        os << EnumClass##Wrapper::ToString(enumClass); \
        return os; \
    } \
    static EnumClass##Wrapper EnumClass##Type; \
    const EnumClass EnumClass##Null = EnumClass::Length; \
    inline constexpr bool IsValid(EnumClass const& enumClass) { return enumClass != EnumClass##Null; }


#define CREATE_ENUM(EnumClass, ...) CREATE_ENUM_COMMON(EnumClass, uint8_t, __VA_ARGS__)
#define CREATE_ENUM_SIGNED(EnumClass, ...) CREATE_ENUM_COMMON(EnumClass, int8_t, __VA_ARGS__)