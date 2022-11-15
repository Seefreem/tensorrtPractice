/* Copyright (c) 1993-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <NvInfer.h>
#include "cudaWrapper.h"
#include "ioHelper.h"
#include <NvOnnxParser.h>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <math.h>

using namespace nvinfer1;
using namespace std;
using namespace cudawrapper;

static Logger gLogger;

constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30; // 1 GB

ICudaEngine* createCudaEngine(string const& onnxModelPath, int batchSize)
{ 
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{nvinfer1::createInferBuilder(gLogger)};
    unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{builder->createNetworkV2(explicitBatch)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};
    unique_ptr<nvinfer1::IBuilderConfig,Destroy<nvinfer1::IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    // builder->setFp16Mode(builder->platformHasFastFp16()); // 和能编译通过的代码文件 就多了这行
    if (builder->platformHasFastFp16())
        config->setFlag(BuilderFlag::kFP16);
    builder->setMaxBatchSize(batchSize);
    
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{32, 3, 256 , 256});    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);   
}

//! \brief 读取已经存在的engine文件，如果不存在，则编译engine文件，并且保存起来
ICudaEngine* getCudaEngine(string const& onnxModelPath, int batchSize)
{
    string enginePath{"engines/" + getBasename(onnxModelPath) + "_batch" + to_string(batchSize) + ".engine"};
    std::cout << "enginePath: " << enginePath << std::endl;
    ICudaEngine* engine{nullptr};

    string buffer = readBuffer(enginePath);
    
    if (buffer.size())
    {
        // Try to deserialize engine.
        std::cout << "Load an engine" << std::endl;
        unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);

    }

    if (!engine)
    {
        // Fallback to creating engine from scratch.
        std::cout << "Build an engine" << std::endl;
        engine = createCudaEngine(onnxModelPath, batchSize);

        if (engine)
        {
            std::cout << "Save an engine" << std::endl;
            unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }
    }
    return engine;
}

int main(int argc, char* argv[])
{
    // Declaring cuda engine.
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    vector<string> inputFiles;

    if (argc < 3)
    {
        cout << "usage: " << argv[0] << " <path_to_model.onnx> <batch size>" << endl;
        return 1;
    }

    string onnxModelPath(argv[1]);
    int batchSize = std::stoi(argv[2]);
    // Create Cuda Engine. 
    engine.reset(getCudaEngine(onnxModelPath, batchSize));
    if (!engine)
        return 1;

    return 0;
}
