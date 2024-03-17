#pragma once
//
// Created by dewe on 3/12/24.
//
#include "torch/torch.h"
#include "options/base.h"
#include "factory/registry.h"


namespace dlb {

    using Tensor2 = std::tuple<torch::Tensor, torch::Tensor>;

    struct ForwardableImpl : public torch::nn::Module {
        ForwardableImpl() = default;

        virtual torch::Tensor forward(const torch::Tensor &x) {
            return x;
        }

        virtual bool has_state() {
            return false;
        }

        virtual const BaseModuleOption *GetOption() const {
            return nullptr;
        }

        virtual void reset_state(int64_t batch_size) {
            AssertIfFalse(has_state(), "ImplementationError: reset_state is not implemented for stateless module");
        }

        virtual std::string GetName() const {
            return name();
        }
    };
    TORCH_MODULE(Forwardable);

    template<class TorchModuleT, class OptionTypeT, const char* class_name>
    class BaseModuleImpl : public ForwardableImpl {

    public:
        using TorchModule = TorchModuleT;
        using OptionType = OptionTypeT;

        BaseModuleImpl(OptionType const &option) : m_option(option) {
            SetInitialized(TorchModule{option.impl});
            register_module(class_name, m_model);
        }

        torch::Tensor forward(const torch::Tensor &x) final {
            return m_model->forward(x);
        }

        bool has_state() final {
            return false;
        }

        const BaseModuleOption *GetOption() const final {
            return &m_option;
        }

        std::string GetName() const final {
            return class_name;
        }

    protected:
        template<class T>
        void Set(T &&module) {
            m_model = std::forward<T>(module);
        }

        template<class T>
        void SetInitialized(T &&module) {
            InitializeWeightBias(std::forward<T>(module), this->m_option);
            Set(module);
        }

    private:
        std::string m_name{};
        TorchModule m_model{nullptr};
        OptionType m_option;
    };

    class BaseSequentialModuleImpl : public ForwardableImpl {

    public:
        BaseSequentialModuleImpl() = default;

        BaseSequentialModuleImpl(std::string parentName,
                                 torch::nn::Sequential model) : m_name(std::move(parentName)),
                                                                m_model(std::move(model)) {
            RegisterModule();
        }

        BaseSequentialModuleImpl(std::string name) : m_name(std::move(name)) {
            RegisterModule();
        }

        torch::Tensor forward(const torch::Tensor &x) noexcept final {
            return m_model->forward(x);
        }

        bool has_state() override
        {
            return true;
        }

        void reset_state(int64_t batch_size) override {
            std::ranges::for_each(m_model->children(),
                                  [batch_size](const std::shared_ptr<torch::nn::Module> &module) {
                                      if (auto fwd = module->as<ForwardableImpl>())
                                      {
                                          if (fwd->has_state())
                                          {
                                              fwd->reset_state(batch_size);
                                          }
                                      }
                                  });
        }

        std::string GetName() const final {
            return m_name;
        }

    protected:
        template<class ModuleType>
        void Append(std::string const &name, torch::nn::ModuleHolder<ModuleType> &&module) {
            m_model->push_back(GetName(name), std::move(module));
        }

        void AppendFlattener() {
            m_model->push_back(GetName("flatten"), torch::nn::Flatten());
        }

        template<class ModuleType>
        void AppendInitialized(std::string const &name,
                               ModuleType &&module, BaseModuleOption const &option) {
            InitializeWeightBias(module, option);
            Append(name, std::forward<ModuleType>(module));
        }

        void AppendActivationFunction(std::optional<ActivationFunction> const &activations) {
            if (activations) {
                m_model->push_back(GetName(ActivationFunctionWrapper::ToString(*activations)),
                                   GetModule(*activations));
            }
        }

        std::string GetName(std::string const &name) {
            ++m_nameCounter[name];
            return m_nameCounter[name] == 1 ? name : fmt::format("{}{}", name, m_nameCounter[name] - 1);
        }

        template<class ModuleType, class OptionType>
        void Fill(std::string const &elementName, OptionType const &options) {
            for (const auto &option: options) {
                AppendInitialized(elementName, ModuleType{option.impl}, option);
                AppendActivationFunction(option.activations);
                if (option.flatten) {
                    AppendFlattener();
                }
            }
        }

        void RegisterModule() {
            register_module(m_name, m_model);
        }

        const torch::nn::Sequential &Models() const {
            return m_model;
        }

    private:
        std::string m_name{};
        torch::nn::Sequential m_model;
        std::unordered_map<std::string, int> m_nameCounter{};
    };

    template<class ClassImpl, class ClassOption>
    Forwardable RegisterModule(YAML::Node const &node, int64_t in_features) {
        ClassOption opt;
        Make(node, in_features, opt);
        std::shared_ptr<ForwardableImpl> base = std::make_shared<ClassImpl>(opt);
        return std::move(base);
    }

    template<class ModuleAliasType>
    Forwardable RegisterTorchModule(YAML::Node const &node, int64_t in_features) {
        std::optional<typename ModuleAliasType::OptionType> opt;
        Make(node, in_features, opt);
        std::shared_ptr<ForwardableImpl> base = std::make_shared<ModuleAliasType>(*opt);
        return std::move(base);
    }
#define REGISTER_SEQ_MODULE(ClassName, ModuleType)\
    class ClassName##Impl : public BaseSequentialModuleImpl {\
\
    public:\
        ClassName##Impl() = default;\
\
        ClassName##Impl(ClassName ## Option const &options) : BaseSequentialModuleImpl(#ClassName), m_options(options) {\
            Fill<torch::nn:: ModuleType>(#ModuleType, options);                                    \
        }\
                                                  \
      const BaseModuleOption* GetOption() const final \
        {\
            return &m_options;\
        }                                            \
    private:                                      \
        ClassName ## Option m_options;\
    };                                                         \
    TORCH_MODULE(ClassName); \
    const size_t ClassName##Impl##RegistrationIndex =          \
    Register(#ClassName, RegisterModule<ClassName ## Impl, ClassName ## Option>)

#define REGISTER_FORWARDABLE_MODULE(BaseClass, ModuleType) \
constexpr char BaseClass##ModuleType##Name[] = #ModuleType;                                                           \
using ModuleType ## Impl = BaseClass<torch::nn:: ModuleType, ModuleType ## Options, BaseClass##ModuleType##Name>; \
TORCH_MODULE(ModuleType);                                  \
const size_t ModuleType##Impl##RegistrationIndex =                                                    \
Register(#ModuleType, RegisterTorchModule<ModuleType ## Impl>)

#define REGISTER_MODULE(ModuleType) REGISTER_FORWARDABLE_MODULE(BaseModuleImpl, ModuleType)

}
