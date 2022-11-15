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
using namespace cudawrapper;

static Logger gLogger;

// Maxmimum absolute tolerance for output tensor comparison against reference.
// 最大绝对误差
constexpr double ABS_EPSILON = 0.005;
// Maxmimum relative tolerance for output tensor comparison against reference.
// 最大相对误差
constexpr double REL_EPSILON = 0.05;
/*
The SimpleOnnx::buildEngine function parses the ONNX model and holds it in the network object. 
To handle the dynamic input dimensions of input images and shape tensors for U-Net model, 
you must create an optimization profile from the builder class, as shown in the following code example.

The number of inputs and outputs, as well as the value and dimension of each, 
can be queried using functions from the ICudaEngine class. 

创建engine文件
*/
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

static int getBindingInputIndex(nvinfer1::IExecutionContext* context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

/*
This application places inference requests on the GPU asynchronously in the function launchInference.
The example uses CUDA streams to manage asynchronous work on the GPU. 
加载数据到GPU内存，执行推理，转移数据到CPU内存
*/
void launchInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
    int inputId = getBindingInputIndex(context);
    // 将数据从CPU内存拷贝到GPU内存
    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    // The inference is then performed with the enqueueV2 function.
    // The enqueueV2 function places inference requests on CUDA streams and takes as input runtime batch size, 
    // pointers to input and output, plus the CUDA stream to be used for kernel execution. 
    context->enqueueV2(bindings, stream, nullptr);
    // 将数据拷贝回CPU内存
    cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

}

void verifyOutput(vector<float> const& outputTensor, vector<float> const& referenceTensor, int size)
{
    for (size_t i = 0; i < size; ++i)
    {
        double reference = static_cast<double>(referenceTensor[i]);
        // Check absolute and relative tolerance.
        if (abs(outputTensor[i] - reference) > max(abs(reference) * REL_EPSILON, ABS_EPSILON))
        {
            cout << "ERROR: mismatch at position " << i;
            cout << " expected " << reference << ", but was " << outputTensor[i] << endl;
            return;
        }
    }
    cout << "OK" << endl;              
}

void saveImageAsPGM(vector<float>& outputTensor,int H, int W)
{
    FILE* pgmimg; 
    pgmimg = fopen("data/output/output.pgm", "wb"); 
  
    fprintf(pgmimg, "P2\n");  
    // Writing Width and Height 
    fprintf(pgmimg, "%d %d\n", H, W);  
    // Writing the maximum gray value 
    fprintf(pgmimg, "255\n");  
    
    for (int i=0;  i< H; ++i)
    {
      for(int j=0; j<W; ++j)
      {
	int temp = round(255* outputTensor[i*H + j]);
        fprintf(pgmimg, "%d ", temp); 
      }
      fprintf(pgmimg, "\n"); 
    }
    
    fclose(pgmimg);
}

int main(int argc, char* argv[])
{
    //================S1: 加载模型
    // Declaring cuda engine.
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    vector<string> inputFiles;

    if (argc < 3)
    {
        cout << "usage: " << argv[0] << " <path_to_model.onnx> <path_to_input.pb>" << endl;
        return 1;
    }
    string onnxModelPath(argv[1]);
    for (int i = 2; i < argc; ++i)
        inputFiles.push_back(string{argv[i]}); 
    int batchSize = inputFiles.size();

    // Create Cuda Engine.
    // 加载engine
    engine.reset(getCudaEngine(onnxModelPath, batchSize));
    string enginePath = onnxModelPath.replace(onnxModelPath.end() - 4,onnxModelPath.end(), "engine");       
    if (!engine)
        return 1;

    //================S2: 分配内存&读取输入数据
    void* bindings[2]{0};
    vector<float> inputTensor;
    vector<float> outputTensor;
    // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
    // getNbBindings() 返回模型的输出+输出的参数数量
    assert(engine->getNbBindings() == 2);
    // bindingIsInput() 返回index指定的参数是不是input
    assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

    // 为每个输入输出参数分配GPU内存
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        Dims dims{engine->getBindingDimensions(i)};
        size_t size = accumulate(dims.d+1, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], batchSize * size * sizeof(float));

        // Resize CPU buffers to fit Tensor.
        if (engine->bindingIsInput(i)){
            inputTensor.resize(size);
        }
        else
            outputTensor.resize(size);
    }

    // Read input tensor from pb file.
    if (readTensor(inputFiles, inputTensor) != inputTensor.size())
    {
        cout << "Couldn't read input Tensor" << endl;
        return 1;
    }
    
    //================S3: 创建执行环境&执行推理
    // Declaring execution context.
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};
    // Create Execution Context.
    // After an engine has been created, create an execution context 
    // to hold intermediate activation values generated during inference. 
    context.reset(engine->createExecutionContext());
    
    Dims dims_i{engine->getBindingDimensions(0)};
    Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
    context->setBindingDimensions(0, inputDims);
    // 实现数据转移和推理
    CudaStream stream;
    launchInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);

    Dims dims{engine->getBindingDimensions(1)};
    saveImageAsPGM(outputTensor, dims.d[2], dims.d[3]);
    // Wait until the work is finished.
    // Using the cudaStreamSynchronize function after calling launchInference 
    // ensures GPU computations complete before the results are accessed. 
    cudaStreamSynchronize(stream);


    //================S4: 验证推理结果
    vector<string> referenceFiles;
    for (string path : inputFiles)
        referenceFiles.push_back(path.replace(path.rfind("input"), 5, "output"));
    // Try to read and compare against reference tensor from protobuf file.

    vector<float> referenceTensor;
    referenceTensor.resize(outputTensor.size());
    if (readTensor(referenceFiles, referenceTensor) != referenceTensor.size())
    {
        cout << "Couldn't read reference Tensor" << endl;
        return 1;
    }

    Dims dims_o{engine->getBindingDimensions(1)};
    int size = batchSize * dims_o.d[2] * dims_o.d[3];
    verifyOutput(outputTensor, referenceTensor, size);
    
    for (void* ptr : bindings)
        cudaFree(ptr);

    return 0;
}


