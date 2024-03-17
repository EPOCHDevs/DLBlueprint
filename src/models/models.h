#pragma once
//
// Created by dewe on 3/12/24.
//
#include "options/options.h"
#include "models/base.h"
#include "models/vision/resnet.h"
#include "models/sequence/recurrent_net.h"

#include "factory/linear.h"
#include "factory/sparse.h"
#include "factory/recurrent.h"
#include "factory/vision.h"


namespace dlb
{
    REGISTER_MODULE(Linear);
    REGISTER_SEQ_MODULE(FCNN, Linear);

//    REGISTER_MODULE(Embedding);
//    REGISTER_SEQ_MODULE(Embeddings, Embedding);

    /***************** Recurrent *************/
    REGISTER_FORWARDABLE_MODULE(RecurrentNetImpl, RNN);
    REGISTER_FORWARDABLE_MODULE(RecurrentNetImpl, GRU);
    REGISTER_FORWARDABLE_MODULE(RecurrentNetImpl, LSTM);
    /**************************************/

    /***************** Conv 2D *************/
//    REGISTER_MODULE(Conv2d);
//    REGISTER_SEQ_MODULE(CNN, Conv2d);
//
//    REGISTER_MODULE(ConvTranspose2d);
//    REGISTER_SEQ_MODULE(CNNTranspose, ConvTranspose2d);
    /*****************************************/

    /****************** RESNET **************/
//    REGISTER_MODULE(BasicBlock);
//    REGISTER_MODULE(Bottleneck);
//
//    using BasicBlockResnetOptions = ResNetOptions;
//    REGISTER_MODULE(BasicBlockResnet);
//
//    using BottleneckResnetOptions = ResNetOptions;
//    REGISTER_MODULE(BottleneckResnet);
//
//    using Resnet18Options = ResNetOptions;
//    REGISTER_MODULE(Resnet18);

//    REGISTER_MODULE(Resnet34);
//    REGISTER_MODULE(Resnet50);
//    REGISTER_MODULE(Resnet101);
//    REGISTER_MODULE(Resnet152);
//    REGISTER_MODULE(Resnext50_32x4d);
//    REGISTER_MODULE(Resnext101_32x8d);
//    REGISTER_MODULE(WideResnet50_2);
//    REGISTER_MODULE(Wide_ResNet101_2);
    /***************************************/

    /****************** FUNCTIONALS **************/
    /*********************************************/

}