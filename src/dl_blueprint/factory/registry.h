#pragma once
//
// Created by dewe on 3/14/24.
//
#include "yaml-cpp/yaml.h"
#include "dl_blueprint/common/defs.h"


namespace dlb {
    using RegistrySignature = std::function<struct Forwardable(YAML::Node const &node, int64_t in_features)>;

    class ForwardableModuleRegistry {

    public:
        ForwardableModuleRegistry(ForwardableModuleRegistry const &) = delete;

        ForwardableModuleRegistry(ForwardableModuleRegistry &&) = delete;

        static ForwardableModuleRegistry &Instance() {
            static ForwardableModuleRegistry instance;
            return instance;
        }


        size_t Add(std::string const &name, RegistrySignature registrySignature) {
            size_t index = m_registry.size();
            if (m_registryIndex.contains(name)) {
                return m_registryIndex[name];
            }
            m_registry.emplace(name, std::move(registrySignature));
            m_registryIndex[name] = index;
            return index;
        }

        RegistrySignature operator()(std::string const &name) const {
            DL_AssertIfTrueF(m_registry.contains(name), "registry does not contain {}.", name);
            return m_registry.at(name);
        }

    private:
        ForwardableModuleRegistry() = default;

        std::unordered_map<std::string, RegistrySignature> m_registry;
        std::unordered_map<std::string, size_t> m_registryIndex;
    };

    inline size_t Register(std::string const &name, RegistrySignature registrySignature) {
        return ForwardableModuleRegistry::Instance().Add(name, std::move(registrySignature));
    }

    static const ForwardableModuleRegistry &MakeModule = ForwardableModuleRegistry::Instance();
}