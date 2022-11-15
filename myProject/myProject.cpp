
#include <NvInfer.h> // ICudaEngine
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h" // nvonnxparser::IParser
#include "include/cudaWrapper.h"
#include "include/ioHelper.h"

#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/macros.h>
#include <google/protobuf/stubs/platform_macros.h> 
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/mutex.h>
#include <google/protobuf/stubs/callback.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <math.h>
#include <cmath>

using namespace nvinfer1;
using namespace std;


static Logger gLogger;

// #define GOOGLE_PROTOBUF_VERIFY_VERSION                                    \
//   ::google::protobuf::internal::VerifyVersion(                            \
//     GOOGLE_PROTOBUF_VERSION, GOOGLE_PROTOBUF_MIN_LIBRARY_VERSION,         \
//     __FILE__)


// std::ostream& operator<<(std::ostream& o, const ILogger::Severity severity);
// class Logger : public nvinfer1::ILogger
// {
// public:
//     virtual void log(Severity severity, const char* msg) noexcept override
//     {
//         std::cerr << severity << ": " << msg << std::endl;
//     }
// };

// static Logger gLogger;

// template <typename T>
// struct Destroy
// {
//     void operator()(T* t) const
//     {
//         t->destroy();
//     }
// };

// void writeBuffer(void* buffer, size_t size, string const& path)
// {
//     ofstream stream(path.c_str(), ios::binary);

//     if (stream)
//         stream.write(static_cast<char*>(buffer), size);
// }


nvinfer1::ICudaEngine* createCudaEngine(string const& onnxModelPath, int batchSize)
{   
    const auto explicitBatch =                                                          
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>>                         
        builder{nvinfer1::createInferBuilder(gLogger)};
    unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>>     
        network{builder->createNetworkV2(explicitBatch)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>>                   
        parser{nvonnxparser::createParser(*network, gLogger)};
    unique_ptr<nvinfer1::IBuilderConfig,Destroy<nvinfer1::IBuilderConfig>>              
        config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }
    /*
    The setMaxBatchSize() function in the following code example is used to 
    specify the maximum batch size that a TensorRT engine expects. 
    
    The setMaxWorkspaceSize() function allows you to increase 
    the GPU memory footprint during the engine building phase.
    */
    builder->setMaxBatchSize(batchSize);
    config->setMaxWorkspaceSize((1 << 30));
    
    /*
    The optimization profile enables you to set the optimum input, minimum, and maximum dimensions to the profile. 
    The builder selects the kernel that results in the lowest runtime for input tensor dimensions 
    and which is valid for all input tensor dimensions in the range between the minimum and maximum dimensions. 
    It also converts the network object into a TensorRT engine.
    */
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{32, 3, 256 , 256});    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

// // returns number of floats successfully read from tensor protobuf
// size_t readTensorProto(string const& path, float* buffer)
// {
//     string const data{readBuffer(path)};
//     onnx::TensorProto tensorProto;
//     if (!tensorProto.ParseFromString(data))
//         return 0;

//     assert(tensorProto.has_raw_data());
//     assert(tensorProto.raw_data().size() % sizeof(float) == 0);

//     memcpy(buffer, tensorProto.raw_data().data(), tensorProto.raw_data().size());
//     return tensorProto.raw_data().size() / sizeof(float);
// }

// // returns number of floats successfully read from tensorProtoPaths
// size_t readTensor(vector<string> const& tensorProtoPaths, vector<float>& buffer)
// {
//     // GOOGLE_PROTOBUF_VERIFY_VERSION;
//     size_t totalElements = 0;

//     for (size_t i = 0; i < tensorProtoPaths.size(); ++i)
//     {
//         size_t elements = readTensorProto(tensorProtoPaths[i], &buffer[totalElements]);        if (!elements)
//         {
//             cout << "ERROR: could not read tensor from file " << tensorProtoPaths[i] << endl;
//             break;
//         }
//         totalElements += elements;
//     }

//     return totalElements;
// }

int main(int argc, char* argv[]){
    
    // 加载模型
    // Declaring cuda engine.
    vector<float> inputTensor;
    vector<float> outputTensor;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};

    string onnxName(argv[1]);
    string enginePath = onnxName.replace(onnxName.end() - 4,onnxName.end(), "engine");  
    
    int batchSize(1);
    engine.reset(createCudaEngine(onnxName, batchSize));
    
    // 编译和存储engine文件
    if (!engine)
        return 1;
    else 
    {
        // 保存engine到文件
        unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{engine->serialize()};
        // Try to save engine for future uses.
        writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
    }

    // // 为每个输入输出参数分配GPU内存
    // for (int i = 0; i < engine->getNbBindings(); ++i)
    // {
    //     Dims dims{engine->getBindingDimensions(i)};
    //     size_t size = accumulate(dims.d+1, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
    //     // Create CUDA buffer for Tensor.
    //     cudaMalloc(&bindings[i], batchSize * size * sizeof(float));

    //     // Resize CPU buffers to fit Tensor.
    //     if (engine->bindingIsInput(i)){
    //         inputTensor.resize(size);
    //     }
    //     else
    //         outputTensor.resize(size);
    // }


    // 加载数据
    // string inputName(argv[2]);
    // inputBuffer;
    // readTensor(inputName, inputBuffer)
    // 执行推理

    // 结果验证
}



