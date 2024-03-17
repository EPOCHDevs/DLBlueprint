//
// Created by dewe on 3/15/24.
//
#include "catch.hpp"
#include "filesystem"
#include "compile.h"
#include "blueprint.h"
#include "models/models.h"

const std::filesystem::path TEST_FILES{__TEST_FILES__};
constexpr int64_t BATCH_SIZE = 256;
constexpr int64_t SEQUENCE_LENGTH = 20;

const dlb::FeatureInput singleDIMInput{{"box", dlb::Shape{250}}};
const dlb::TensorDict singleDIMTensor{{"box", torch::ones({BATCH_SIZE, SEQUENCE_LENGTH, 250})}};


TEST_CASE("Training files/actor_critic_gru.yaml")
{
    const YAML::Node config = YAML::LoadFile(TEST_FILES / "actor_critic_gru.yaml");
    dlb::Blueprint blueprint = dlb::Build(singleDIMInput, config);
    blueprint.reset_state(BATCH_SIZE);

    auto compiledModule = blueprint.named_children();
    REQUIRE(compiledModule.size() == 3);

    std::shared_ptr<torch::nn::Module> &shared = compiledModule["shared"];
    std::shared_ptr<torch::nn::Module> &actor = compiledModule["actor"];
    std::shared_ptr<torch::nn::Module> &critic = compiledModule["critic"];

    SECTION("Verify Forward")
    {
        dlb::TensorDict output = singleDIMTensor;
        blueprint.forward(output);

        REQUIRE(output[0].key() == "box");
        REQUIRE(output[1].key() == "shared");
        REQUIRE(output[3].key() == "actor");
        REQUIRE(output[2].key() == "critic");

        REQUIRE(output["shared"].sizes() == std::vector<int64_t>{BATCH_SIZE, 128});
        REQUIRE(output["actor"].sizes() == std::vector<int64_t>{BATCH_SIZE, 2});
        REQUIRE(output["critic"].sizes() == std::vector<int64_t>{BATCH_SIZE, 1});
    }
}


//TEST_CASE("Validate files/linear/basic.yaml")
//{
//    const std::vector<YAML::Node> config = YAML::LoadAllFromFile(TEST_FILES / "linear/basic.yaml");
//
//    SECTION("Scalar arguments")
//    {
//        auto compiledModule = dlb::Build({250}, config[0]);
//
//        REQUIRE(compiledModule.size() == 1);
//        REQUIRE(compiledModule.contains("top"));
//
//        auto &topModule = compiledModule.at("top");
//        REQUIRE(topModule->size() == 1);
//
//        const auto &fcnnWrapper = topModule->at<dlb::ForwardableImpl>(0);
//        REQUIRE(fcnnWrapper.GetName() == "FCNN");
//
//        const auto fcnn = fcnnWrapper.as<dlb::FCNNImpl>()->named_children();
//        REQUIRE(fcnn.size() == 1);
//
//        const auto fcnnModules = fcnn["FCNN"]->named_children();
//        REQUIRE(fcnnModules.size() == 4);
//
//        auto layer0 = fcnnModules[0];
//        auto fc0 = layer0.value()->as<torch::nn::Linear>();
//        REQUIRE(layer0.key() == "Linear");
//        REQUIRE(fc0->options.in_features() == 250);
//        REQUIRE(fc0->options.out_features() == 64);
//        REQUIRE(fc0->options.bias() == true);
//
//        auto layer1 = fcnnModules[1];
//        REQUIRE(layer1.key() == "tanh");
//        REQUIRE(layer1.value());
//
//        auto layer2 = fcnnModules[2];
//        REQUIRE(layer2.key() == "Linear1");
//        auto fc1 = layer2.value()->as<torch::nn::Linear>();
//        REQUIRE(fc1->options.in_features() == 64);
//        REQUIRE(fc1->options.out_features() == 64);
//        REQUIRE(fc1->options.bias() == true);
//
//        auto layer3 = fcnnModules[3];
//        REQUIRE(layer3.key() == "tanh1");
//        REQUIRE(layer3.value());
//    }
//
//    SECTION("Sequential Arguments")
//    {
//        auto compiledModule = dlb::Build({250}, config[1]);
//
//        REQUIRE(compiledModule.size() == 1);
//        REQUIRE(compiledModule.contains("top"));
//
//        auto &topModule = compiledModule.at("top");
//        REQUIRE(topModule->size() == 1);
//
//        const auto &fcnnWrapper = topModule->at<dlb::ForwardableImpl>(0);
//        REQUIRE(fcnnWrapper.GetName() == "FCNN");
//
//        const auto fcnn = fcnnWrapper.as<dlb::BaseSequentialModuleImpl>()->named_children();
//        REQUIRE(fcnn.size() == 1);
//
//        const auto fcnnModules = fcnn["FCNN"]->named_children();
//        REQUIRE(fcnnModules.size() == 4);
//
//        auto layer0 = fcnnModules[0];
//        auto fc0 = layer0.value()->as<torch::nn::Linear>();
//        REQUIRE(layer0.key() == "Linear");
//        REQUIRE(fc0->options.in_features() == 250);
//        REQUIRE(fc0->options.out_features() == 64);
//        REQUIRE(fc0->options.bias() == true);
//
//        auto layer1 = fcnnModules[1];
//        REQUIRE(layer1.key() == "sigmoid");
//        REQUIRE(layer1.value());
//
//        auto layer2 = fcnnModules[2];
//        REQUIRE(layer2.key() == "Linear1");
//        auto fc1 = layer2.value()->as<torch::nn::Linear>();
//        REQUIRE(fc1->options.in_features() == 64);
//        REQUIRE(fc1->options.out_features() == 32);
//        REQUIRE(fc1->options.bias() == false);
//
//        auto layer3 = fcnnModules[3];
//        REQUIRE(layer3.key() == "tanh");
//        REQUIRE(layer3.value());
//    }
//
//    SECTION("Invalid Mixed Arguments")
//    {
//        REQUIRE_THROWS(dlb::Build({250}, config[2]));
//    }
//}
//
//TEST_CASE("Validate files/linear/rl.yaml")
//{
//    const YAML::Node config = YAML::LoadFile(TEST_FILES / "linear/rl.yaml");
//    auto compiledModule = dlb::Build({250}, config);
//
//    REQUIRE(compiledModule.size() == 3);
//    auto shared = compiledModule["shared"];
//    auto actor = compiledModule["actor"];
//    auto critic = compiledModule["critic"];
//
//}
//
//TEST_CASE("Validate files/recurrent.yaml")
//{
//    const std::array<std::string, 3> types{"GRU", "RNN", "LSTM"};
//    const std::vector<YAML::Node> configs = YAML::LoadAllFromFile(TEST_FILES / "recurrent.yaml");
//
//    auto TEST_RECURRENT_NET_OPTION = [](auto &&recurrent) {
//        REQUIRE(recurrent->options.input_size() == 250);
//        REQUIRE(recurrent->options.hidden_size() == 64);
//        REQUIRE(recurrent->options.bias() == true);
//        REQUIRE(recurrent->options.batch_first() == true);
//        REQUIRE(recurrent->options.bidirectional() == false);
//        REQUIRE(recurrent->options.dropout() == false);
//        REQUIRE(recurrent->options.num_layers() == 1);
//    };
//
//    for (auto &&[type, config]: ranges::view::zip(types, configs)) {
//        DYNAMIC_SECTION(type)
//        {
//            auto compiledModule = dlb::Build({250}, config);
//
//            REQUIRE(compiledModule.size() == 1);
//            REQUIRE(compiledModule.contains("top"));
//
//            auto &topModule = compiledModule.at("top");
//            REQUIRE(topModule->size() == 1);
//
//            const auto &subModule = topModule->at<dlb::ForwardableImpl>(0);
//            REQUIRE(subModule.GetName() == type);
//
//
//            if (type == "GRU") {
//                const auto subModuleChildren = subModule.as<dlb::GRU>()->named_children();
//                REQUIRE(subModuleChildren.size() == 1);
//                TEST_RECURRENT_NET_OPTION(subModuleChildren[type]->as<torch::nn::GRU>());
//            } else if (type == "LSTM") {
//                const auto subModuleChildren = subModule.as<dlb::LSTM>()->named_children();
//                REQUIRE(subModuleChildren.size() == 1);
//                TEST_RECURRENT_NET_OPTION(subModuleChildren[type]->as<torch::nn::LSTM>());
//            } else {
//                const auto subModuleChildren = subModule.as<dlb::RNN>()->named_children();
//                REQUIRE(subModuleChildren.size() == 1);
//                TEST_RECURRENT_NET_OPTION(subModuleChildren[type]->as<torch::nn::RNN>());
//            }
//        }
//    }
//}