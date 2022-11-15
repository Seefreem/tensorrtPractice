/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! sampleINT8API.cpp
//! This file contains implementation showcasing usage of INT8 calibration and precision APIs.
//! It creates classification networks such as mobilenet, vgg19, resnet-50 from onnx model file.
//! This sample showcae setting per-tensor dynamic range overriding calibrator generated scales if it exists.
//! This sample showcase how to set computation precision of layer. It involves forcing output tensor type of the layer
//! to particular precision. It can be run with the following command line: Command: ./sample_int8_api [-h or --help]
//! [-m modelfile] [-s per_tensor_dynamic_range_file] [-i image_file] [-r reference_file] [-d path/to/data/dir]
//! [--verbose] [-useDLA <id>]

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "argsParser.h" 

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_int8_api";

struct SampleINT8APIPreprocessing
{
    // Preprocessing values are available here:
    // https://github.com/onnx/models/tree/master/models/image_classification/resnet
    std::vector<int> inputDims{1, 3, 224, 224};
};

//!
//! \brief The SampleINT8APIParams structure groups the additional parameters required by
//!         the INT8 API sample
//! \brief 这个数据结构例举了使用int8精度所需要的参数
struct SampleINT8APIParams
{
    bool verbose{false};                    // 日志标志位
    bool writeNetworkTensors{false};        // 
    int dlaCore{-1};                        // 

    SampleINT8APIPreprocessing mPreproc;    // 
    std::string modelFileName;              // 模型文件名
    std::vector<std::string> dataDirs;      // 
    std::string dynamicRangeFileName;       // 模型每一层的张量的动态范围配置文件
    std::string imageFileName;              // 输入图像文件名
    std::string referenceFileName;          // 用于验证量化模型精度的参考文件
    std::string networkTensorsFileName;     // 保存模型中张量名字的文件
    /* For example:
    std::string modelFileName{"resnet50.onnx"};
    std::string imageFileName{"airliner.ppm"};
    std::string referenceFileName{"reference_labels.txt"};
    std::string dynamicRangeFileName{"resnet50_per_tensor_dynamic_range.txt"};
    std::string networkTensorsFileName{"network_tensors.txt"};
    */
};

//!
//! \brief The SampleINT8API class implements INT8 inference on classification networks.
//!
//! \details INT8 API usage for setting custom int8 range for each input layer. API showcase how
//!           to perform INT8 inference without calibration table
//!         展示了如何使用int8进行推理，并且没有使用校准表
class SampleINT8API
{
private:
    // 析构器
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleINT8API(const SampleINT8APIParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!         创建模型对象、builder等，读取ONNX文件，build engine，获取输入输出维度信息
    sample::Logger::TestResult build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!         创建执行环境，创建流，拷贝数据，执行推理，等待推理结束
    sample::Logger::TestResult infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //! 清空变量
    sample::Logger::TestResult teardown();
    // 要使用到的参数
    SampleINT8APIParams mParams; //!< Stores Sample Parameter

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network
    // 存放输入输出张量的名字(mapping)
    std::map<std::string, std::string> mInOut; //!< Input and output mapping of the network

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network
    // 对于每一层的动态范围
    std::unordered_map<std::string, float>
        mPerTensorDynamicRangeMap; //!< Mapping from tensor name to max absolute dynamic range values
    //          获取模型输入输出张量的名字
    void getInputOutputNames(); //!< Populates input and output mapping of the network

    //!
    //! \brief Reads the ppm input image, preprocesses, and stores the result in a managed buffer
    //!         这个函数做了三件事：读取ppm图片，将图片从HWC转化为CHW，并且标准化到[-1, 1]
    //!         最终将数据存放在 buffers 中
    bool prepareInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Verifies that the output is correct and prints it
    //!         这里的验证并不是指和某一个参考输出进行比较，而是指将量化后的模型的输出结果打印出来看
    bool verifyOutput(const samplesCommon::BufferManager& buffers) const;

    //!
    //! \brief Populate per-tensor dynamic range values
    //!         读取模型中的每一个张量的动态范围（取值范围的绝对值的最大值），用于量化
    //!         存储在 mPerTensorDynamicRangeMap中
    bool readPerTensorDynamicRangeValues();

    //!
    //! \brief  Sets custom dynamic range for network tensors
    //!         调用了 readPerTensorDynamicRangeValues 
    //!         这个函数读取了每一层的动态范围的文件，然后将动态范围的值设置到每一层中
    bool setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief  Sets computation precision for network layers
    //!         为模型的每一个可计算层设置精度（int8），为每一层的可计算输出张量设置精度
    void setLayerPrecision(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief  Write network tensor names to a file.
    //!         遍历模型中的张量，将张量的名字写入文件。注意模型中的张量就是模型的输入张量和模型中的常数张量，还有每一层的输出张量
    //!         模型的输出就是模型的最后一层的输出。因此在遍历模型的张量的时候，只用关注模型的输入张量和每一层的输出张量
    void writeNetworkTensorNames(const SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
};

//!
//! \brief  Populates input and output mapping of the network
//!         获取模型输入输出张量的名字
void SampleINT8API::getInputOutputNames()
{
    // 获取输入输出tensor个数
    int nbindings = mEngine.get()->getNbBindings();
    ASSERT(nbindings == 2);
    for (int b = 0; b < nbindings; ++b)
    {
        // 获取tensor的维度信息
        nvinfer1::Dims dims = mEngine.get()->getBindingDimensions(b);
        if (mEngine.get()->bindingIsInput(b))
        {
            // 打印相关信息
            if (mParams.verbose)
            {
                sample::gLogInfo << "Found input: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                                 << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            // 保存输入张量名
            mInOut["input"] = mEngine.get()->getBindingName(b);
        }
        else // output tensor
        {
            if (mParams.verbose)
            {
                sample::gLogInfo << "Found output: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                                 << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            // 保存输出张量名
            mInOut["output"] = mEngine.get()->getBindingName(b);
        }
    }
}

//!
//! \brief Populate per-tensor dyanamic range values
//!         读取模型中的每一个张量的动态范围（取值范围的绝对值的最大值），用于量化
bool SampleINT8API::readPerTensorDynamicRangeValues()
{
    std::ifstream iDynamicRangeStream(mParams.dynamicRangeFileName);
    if (!iDynamicRangeStream)
    {
        sample::gLogError << "Could not find per-tensor scales file: " << mParams.dynamicRangeFileName << std::endl;
        return false;
    }

    std::string line;
    char delim = ':'; // “换行符”
    while (std::getline(iDynamicRangeStream, line))
    {
        std::istringstream iline(line);
        std::string token;
        // use separator to read parts of the line
        // 按照delim中的字符将iline中的字符串分割，并将第一个被分割出的元素放在token中，
        // 每次分割之后iline就变短一些。
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        // std::stof(token): string to float
        float dynamicRange = std::stof(token);
        mPerTensorDynamicRangeMap[tensorName] = dynamicRange;
    }
    return true;
}

//!
//! \brief  Sets computation precision for network layers
//!         为模型的每一个可计算层设置精度（int8），为每一层的可计算输出张量设置精度
void SampleINT8API::setLayerPrecision(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    sample::gLogInfo << "Setting Per Layer Computation Precision" << std::endl;
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);
        if (mParams.verbose)
        {
            std::string layerName = layer->getName();
            sample::gLogInfo << "Layer: " << layerName << ". Precision: INT8" << std::endl;
        }

        // Don't set the precision on non-computation layers as they don't support
        // int8.
        // 注意不要给非计算层设置精度
        if (layer->getType() != LayerType::kCONSTANT && layer->getType() != LayerType::kCONCATENATION
            && layer->getType() != LayerType::kSHAPE)
        {
            // set computation precision of the layer
            // 为每一个计算层设置精度为int8
            layer->setPrecision(nvinfer1::DataType::kINT8);
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            if (mParams.verbose)
            {
                std::string tensorName = layer->getOutput(j)->getName();
                sample::gLogInfo << "Tensor: " << tensorName << ". OutputType: INT8" << std::endl;
            }
            // set output type of execution tensors and not shape tensors.
            // 为当前层可计算的输出tensor设置精度
            if (layer->getOutput(j)->isExecutionTensor())
            {
                layer->setOutputType(j, nvinfer1::DataType::kINT8);
            }
        }
    }
}

//!
//! \brief  Write network tensor names to a file.
//!         遍历模型中的张量，将张量的名字写入文件。注意模型中的张量就是模型的输入张量和模型中的常数张量，还有每一层的输出张量
//!         模型的输出就是模型的最后一层的输出。因此在遍历模型的张量的时候，只用关注模型的输入张量和每一层的输出张量
void SampleINT8API::writeNetworkTensorNames(const SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    sample::gLogInfo << "Sample requires to run with per-tensor dynamic range." << std::endl;
    sample::gLogInfo << "In order to run Int8 inference without calibration, user will need to provide dynamic range for all "
                "the network tensors."
             << std::endl;

    std::ofstream tensorsFile{mParams.networkTensorsFileName};

    // Iterate through network inputs to write names of input tensors.
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        std::string tName = network->getInput(i)->getName();
        tensorsFile << "TensorName: " << tName << std::endl;
        if (mParams.verbose)
        {
            sample::gLogInfo << "TensorName: " << tName << std::endl;
        }
    }

    // Iterate through network layers.
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        // Write output tensors of a layer to the file.
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            std::string tName = network->getLayer(i)->getOutput(j)->getName();
            tensorsFile << "TensorName: " << tName << std::endl;
            if (mParams.verbose)
            {
                sample::gLogInfo << "TensorName: " << tName << std::endl;
            }
        }
    }
    tensorsFile.close();
    sample::gLogInfo << "Successfully generated network tensor names. Writing: " << mParams.networkTensorsFileName
                     << std::endl;
    sample::gLogInfo
        << "Use the generated tensor names file to create dynamic range file for Int8 inference. Follow README.md "
           "for instructions to generate dynamic_ranges.txt file."
        << std::endl;
}

//!
//! \brief  Sets custom dynamic range for network tensors
//!         这个函数读取了每一层的动态范围的文件，然后将动态范围的值设置到每一层中
bool SampleINT8API::setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // populate per-tensor dynamic range
    if (!readPerTensorDynamicRangeValues())
    {
        return false;
    }

    sample::gLogInfo << "Setting Per Tensor Dynamic Range" << std::endl;
    if (mParams.verbose)
    {
        sample::gLogInfo << "If dynamic range for a tensor is missing, TensorRT will run inference assuming dynamic range for "
                    "the tensor as optional."
                 << std::endl;
        sample::gLogInfo << "If dynamic range for a tensor is required then inference will fail. Follow README.md to generate "
                    "missing per-tensor dynamic range."
                 << std::endl;
    }
    // set dynamic range for network input tensors
    // 设置输入向量的动态范围
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        // 获取输入张量名；获取网络的输入层
        std::string tName = network->getInput(i)->getName();
        if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
        {
            // 设置张量动态范围，（min， max）
            if (!network->getInput(i)->setDynamicRange(
                    -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName)))
            {
                return false;
            }
        }
        else
        {
            if (mParams.verbose)
            {
                sample::gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
            }
        }
    }

    // set dynamic range for layer output tensors
    // 为每一层的输出张量设置动态范围；获取网络层数
    // 注意因为每一层也是一个深度学习模型（递归），所以每一层也是有输入输出的。
    // 只是习惯上，将整个模型的输入单独处理，然后将每一层视为只包含层和层的输出
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        // 获取网络的某一层
        auto lyr = network->getLayer(i);
        // 每一层可能有多个输出
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
        {
            // 获取输出的名字
            std::string tName = lyr->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                // 校准器计算出的动态范围可以被下面的这个函数setDynamicRang 覆盖掉
                if (!lyr->getOutput(j)->setDynamicRange(
                        -mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName)))
                {
                    return false;
                }
            }
            else if (lyr->getType() == LayerType::kCONSTANT) // 如果是常数层，那就找到这个常数层中绝对值最大的值，将其作为动态范围
            {
                IConstantLayer* cLyr = static_cast<IConstantLayer*>(lyr);
                if (mParams.verbose)
                {
                    sample::gLogWarning << "Computing missing dynamic range for tensor, " << tName << ", from weights."
                                        << std::endl;
                }
                auto wts = cLyr->getWeights();
                double max = std::numeric_limits<double>::min();
                for (int64_t wb = 0, we = wts.count; wb < we; ++wb)
                {
                    double val{};
                    switch (wts.type)
                    {
                    case DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                    case DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                    case DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                    case DataType::kHALF: val = static_cast<const half_float::half*>(wts.values)[wb]; break;
                    case DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                    case DataType::kUINT8: val = static_cast<uint8_t const*>(wts.values)[wb]; break;
                    }
                    max = std::max(max, std::abs(val));
                }

                if (!lyr->getOutput(j)->setDynamicRange(-max, max))
                {
                    return false;
                }
            }
            else
            {
                if (mParams.verbose)
                {
                    sample::gLogWarning << "Missing dynamic range for tensor: " << tName << std::endl;
                }
            }
        }
    }

    if (mParams.verbose)
    {
        sample::gLogInfo << "Per Tensor Dynamic Range Values for the Network:" << std::endl;
        for (auto iter = mPerTensorDynamicRangeMap.begin(); iter != mPerTensorDynamicRangeMap.end(); ++iter)
            sample::gLogInfo << "Tensor: " << iter->first << ". Max Absolute Dynamic Range: " << iter->second
                             << std::endl;
    }
    return true;
}

//!
//! \brief Preprocess inputs and allocate host/device input buffers
//! 这个函数做了三件事：读取ppm图片，将图片从HWC转化为CHW，并且标准化到[-1, 1]
//! 最终将数据存放在 buffers 中
bool SampleINT8API::prepareInput(const samplesCommon::BufferManager& buffers)
{
    // 获取文件名后缀，检查是否是ppm数据。
    // PPM是最简单的彩色图像格式。 PPM文件是使用文本格式设置的24位彩色图像， 为每个像素分配一个从0到65536的数字，该数字定义了像素的颜色。 
    if (samplesCommon::toLower(samplesCommon::getFileType(mParams.imageFileName)).compare("ppm") != 0)
    {
        sample::gLogError << "Wrong format: " << mParams.imageFileName << " is not a ppm file." << std::endl;
        return false;
    }

    int channels = mParams.mPreproc.inputDims.at(1);
    int height = mParams.mPreproc.inputDims.at(2);
    int width = mParams.mPreproc.inputDims.at(3);
    int max{0};
    std::string magic{""};

    std::vector<uint8_t> fileData(channels * height * width);

    std::ifstream infile(mParams.imageFileName, std::ifstream::binary);
    ASSERT(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(fileData.data()), width * height * channels);

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(mInOut["input"]));

    // Convert HWC to CHW and Normalize
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = c * height * width + h * width + w;
                int srcIdx = h * width * channels + w * channels + c;
                // This equation include 3 steps
                // 1. Scale Image to range [0.f, 1.0f]
                // 2. Normalize Image using per channel Mean and per channel Standard Deviation
                // 3. Shuffle HWC to CHW form
                // 原始数据应该是0-255的
                // 也就是将原始图片数据乘以2，然后再除以255，将数据映射到[0, 2]，然后减去1，得到标准化的数据
                hostInputBuffer[dstIdx] = (2.0 / 255.0) * static_cast<float>(fileData[srcIdx]) - 1.0;
            }
        }
    }
    return true;
}

//!
//! \brief Verifies that the output is correct and prints it
//!        这里的验证并不是指和某一个参考输出进行比较，而是指将量化后的模型的输出结果打印出来看
bool SampleINT8API::verifyOutput(const samplesCommon::BufferManager& buffers) const
{
    // copy output host buffer data for further processing
    const float* probPtr = static_cast<const float*>(buffers.getHostBuffer(mInOut.at("output")));
    std::vector<float> output(probPtr, probPtr + mOutputDims.d[1]);

    auto inds = samplesCommon::argMagnitudeSort(output.cbegin(), output.cend());

    // read reference lables to generate prediction lables
    std::vector<std::string> referenceVector;
    if (!samplesCommon::readReferenceFile(mParams.referenceFileName, referenceVector))
    {
        sample::gLogError << "Unable to read reference file: " << mParams.referenceFileName << std::endl;
        return false;
    }

    std::vector<std::string> top5Result = samplesCommon::classify(referenceVector, output, 5);

    sample::gLogInfo << "SampleINT8API result: Detected:" << std::endl;
    for (int i = 1; i <= 5; ++i)
    {
        sample::gLogInfo << "[" << i << "]  " << top5Result[i - 1] << std::endl;
    }

    return true;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates INT8 classification network by parsing the onnx model and builds
//!          the engine that will be used to run INT8 inference (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!         创建模型对象、builder等，读取ONNX文件，build engine，获取输入输出维度信息
sample::Logger::TestResult SampleINT8API::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        sample::gLogError << "Unable to create builder object." << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }

    // 检测平台是否支持int8推理模型
    if (!builder->platformHasFastInt8())
    {
        sample::gLogError << "Platform does not support INT8 inference. sampleINT8API can only run in INT8 Mode." << std::endl;
        return sample::Logger::TestResult::kWAIVED;
    }

    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH: 
    // Deprecated. This flag has no effect now, but is only kept for backward compatability.
    // 没啥用，单纯后向兼容
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 用互斥参数声明一个模型对象
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        sample::gLogError << "Unable to create network object." << mParams.referenceFileName << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }
    // 创建配置信息对象
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError << "Unable to create config object." << mParams.referenceFileName << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }
    // 创建解析器对象，解析onnx文件，反序列化onnx模型
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        sample::gLogError << "Unable to create parser object." << mParams.referenceFileName << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }

    // Parse ONNX model file to populate TensorRT INetwork
    // 解析onnx文件，并将模型的结构和参数信息写入 network 对象中
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    if (!parser->parseFromFile(mParams.modelFileName.c_str(), verbosity))
    {
        sample::gLogError << "Unable to parse ONNX model file: " << mParams.modelFileName << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }
    // 将模型中的张量的名字写进文件
    if (mParams.writeNetworkTensors)
    {
        writeNetworkTensorNames(network);
        return sample::Logger::TestResult::kWAIVED;
    }

    // Configure buider
    // !< Enable layers marked to execute on GPU if layer cannot execute on DLA.
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    // Enable INT8 model. Required to set custom per-tensor dynamic range or INT8 Calibration
    // 设置模型编译为int8精度，要求先设置好每一个模型张量的动态范围
    config->setFlag(BuilderFlag::kINT8);
    // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
    // 将校准器置空，因为手动设置/显示设置了每一层的动态范围
    config->setInt8Calibrator(nullptr);

    // force layer to execute with required precision
    // 设置每一层的精度
    setLayerPrecision(network);

    // set INT8 Per Tensor Dynamic range
    // 设置每个张量的动态范围
    if (!setDynamicRange(network))
    {
        sample::gLogError << "Unable to set per-tensor dynamic range." << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }

    // CUDA stream used for profiling by the builder.
    // 申明 CUDA stream 用于模型性能分析
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return sample::Logger::TestResult::kFAILED;
    }
    // Set the cuda stream that is used to profile this network.
    config->setProfileStream(*profileStream);
    // build了一个序列化的模型，返回模型指针，但是并不build engine
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        sample::gLogError << "Unable to build serialized plan." << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }
    // Create an instance of an IRuntime class for running engine.
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        sample::gLogError << "Unable to create runtime." << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }

    // build TRT engine
    // build 一个engine，这里展示了另一种build engine的方法。和之前的方法有什么差别？
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        sample::gLogError << "Unable to build cuda engine." << std::endl;
        return sample::Logger::TestResult::kFAILED;
    }

    // populates input output map structure
    getInputOutputNames();

    // derive input/output dims from engine bindings
    // 获取输入输出的维度
    const int inputIndex = mEngine.get()->getBindingIndex(mInOut["input"].c_str());
    mInputDims = mEngine.get()->getBindingDimensions(inputIndex);

    const int outputIndex = mEngine.get()->getBindingIndex(mInOut["output"].c_str());
    mOutputDims = mEngine.get()->getBindingDimensions(outputIndex);

    return sample::Logger::TestResult::kRUNNING;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output
//!         创建执行环境，创建流，拷贝数据，执行推理，等待推理结束
sample::Logger::TestResult SampleINT8API::infer()
{
    // Create RAII buffer manager object
    // BufferManager 负责主机和设备的内存管理
    samplesCommon::BufferManager buffers(mEngine);
    // 创建运行环境
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return sample::Logger::TestResult::kFAILED;
    }

    // Read the input data into the managed buffers
    // There should be just 1 input tensor

    if (!prepareInput(buffers))
    {
        return sample::Logger::TestResult::kFAILED;
    }

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueueV2(buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return sample::Logger::TestResult::kFAILED;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    // Release stream
    CHECK(cudaStreamDestroy(stream));

    // Check and print the output of the inference
    return verifyOutput(buffers) ? sample::Logger::TestResult::kRUNNING : sample::Logger::TestResult::kFAILED;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
sample::Logger::TestResult SampleINT8API::teardown()
{
    return sample::Logger::TestResult::kRUNNING;
}

//!
//! \brief The SampleINT8APIArgs structures groups the additional arguments required by
//!         the INT8 API sample
//!
struct SampleINT8APIArgs : public samplesCommon::Args
{
    bool verbose{false};
    bool writeNetworkTensors{false};
    std::string modelFileName{"resnet50.onnx"};
    std::string imageFileName{"airliner.ppm"};
    std::string referenceFileName{"reference_labels.txt"};
    std::string dynamicRangeFileName{"resnet50_per_tensor_dynamic_range.txt"};
    std::string networkTensorsFileName{"network_tensors.txt"};
};


/*
读取命令行参数分为了三个步骤：1 读入并解析命令行参数；2 将命令行参数写到模型使用的结构体中；3 检查参数是否合理。
*/
//! \brief This function parses arguments specific to SampleINT8API
//!     解析命令行参数，并设置模型相关参数：SampleINT8APIArgs
bool parseSampleINT8APIArgs(SampleINT8APIArgs& args, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--model=", 8))
        {
            args.modelFileName = (argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--image=", 8))
        {
            args.imageFileName = (argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--reference=", 12))
        {
            args.referenceFileName = (argv[i] + 12);
        }
        else if (!strncmp(argv[i], "--write_tensors", 15))
        {
            args.writeNetworkTensors = true;
        }
        else if (!strncmp(argv[i], "--network_tensors_file=", 23))
        {
            args.networkTensorsFileName = (argv[i] + 23);
        }
        else if (!strncmp(argv[i], "--ranges=", 9))
        {
            args.dynamicRangeFileName = (argv[i] + 9);
        }
        else if (!strncmp(argv[i], "--int8", 6))
        {
            args.runInInt8 = true;
        }
        else if (!strncmp(argv[i], "--fp16", 6))
        {
            args.runInFp16 = true;
        }
        else if (!strncmp(argv[i], "--useDLACore=", 13))
        {
            args.useDLACore = std::stoi(argv[i] + 13);
        }
        else if (!strncmp(argv[i], "--data=", 7))
        {
            std::string dirPath = (argv[i] + 7);
            if (dirPath.back() != '/')
            {
                dirPath.push_back('/');
            }
            args.dataDirs.push_back(dirPath);
        }
        else if (!strncmp(argv[i], "--verbose", 9) || !strncmp(argv[i], "-v", 2))
        {
            args.verbose = true;
        }
        else if (!strncmp(argv[i], "--help", 6) || !strncmp(argv[i], "-h", 2))
        {
            args.help = true;
        }
        else
        {
            sample::gLogError << "Invalid Argument: " << argv[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 检查相关的文件是否存在，并且记录对应的文件路径
void validateInputParams(SampleINT8APIParams& params)
{
    sample::gLogInfo << "Please follow README.md to generate missing input files." << std::endl;
    sample::gLogInfo << "Validating input parameters. Using following input files for inference." << std::endl;
    params.modelFileName = locateFile(params.modelFileName, params.dataDirs);
    sample::gLogInfo << "    Model File: " << params.modelFileName << std::endl;
    if (params.writeNetworkTensors)
    {
        sample::gLogInfo << "    Writing Network Tensors File to: " << params.networkTensorsFileName << std::endl;
        return;
    }
    params.imageFileName = locateFile(params.imageFileName, params.dataDirs);
    sample::gLogInfo << "    Image File: " << params.imageFileName << std::endl;
    params.referenceFileName = locateFile(params.referenceFileName, params.dataDirs);
    sample::gLogInfo << "    Reference File: " << params.referenceFileName << std::endl;
    params.dynamicRangeFileName = locateFile(params.dynamicRangeFileName, params.dataDirs);
    sample::gLogInfo << "    Dynamic Range File: " << params.dynamicRangeFileName << std::endl;
    return;
}

//!
//! \brief This function initializes members of the params struct using the command line args
//! 通过命令行输入的参数和内置的默认参数初始化模型的相关参数
SampleINT8APIParams initializeSampleParams(SampleINT8APIArgs args)
{
    SampleINT8APIParams params;
    // args.dataDirs 使用户执行的查找配置文件、模型文件等的路径
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/samples/int8_api/");
        params.dataDirs.push_back("data/int8_api/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.dataDirs.push_back(""); // In case of absolute path search
    params.verbose = args.verbose;
    params.modelFileName = args.modelFileName;
    params.imageFileName = args.imageFileName;
    params.referenceFileName = args.referenceFileName;
    params.dynamicRangeFileName = args.dynamicRangeFileName;
    params.dlaCore = args.useDLACore;
    params.writeNetworkTensors = args.writeNetworkTensors;
    params.networkTensorsFileName = args.networkTensorsFileName;
    validateInputParams(params);
    return params;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_int8_api [-h or --help] [--model=model_file] "
                 "[--ranges=per_tensor_dynamic_range_file] [--image=image_file] [--reference=reference_file] "
                 "[--data=/path/to/data/dir] [--useDLACore=<int>] [-v or --verbose]\n";
    std::cout << "-h or --help. Display This help information" << std::endl;
    std::cout << "--model=model_file.onnx or /absolute/path/to/model_file.onnx. Generate model file using README.md in "
                 "case it does not exists. Default to resnet50.onnx"
              << std::endl;
    std::cout << "--image=image.ppm or /absolute/path/to/image.ppm. Image to infer. Defaults to airlines.ppm"
              << std::endl;
    std::cout << "--reference=reference.txt or /absolute/path/to/reference.txt. Reference labels file. Defaults to "
                 "reference_labels.txt"
              << std::endl;
    std::cout << "--ranges=ranges.txt or /absolute/path/to/ranges.txt. Specify custom per-tensor dynamic range for the "
                 "network. Defaults to resnet50_per_tensor_dynamic_range.txt"
              << std::endl;
    std::cout << "--write_tensors. Option to generate file containing network tensors name. By default writes to "
                 "network_tensors.txt file. To provide user defined file name use additional option "
                 "--network_tensors_file. See --network_tensors_file option usage for more detail."
              << std::endl;
    std::cout << "--network_tensors_file=network_tensors.txt or /absolute/path/to/network_tensors.txt. This option "
                 "needs to be used with --write_tensors option. Specify file name (will write to current execution "
                 "directory) or absolute path to file name to write network tensor names file. Dynamic range "
                 "corresponding to each network tensor is required to run the sample. Defaults to network_tensors.txt"
              << std::endl;
    std::cout << "--data=/path/to/data/dir. Specify data directory to search for above files in case absolute paths to "
                 "files are not provided. Defaults to data/samples/int8_api/ or data/int8_api/"
              << std::endl;
    std::cout << "--useDLACore=N. Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--verbose. Outputs per-tensor dynamic range and layer precision info for the network" << std::endl;
}

int main(int argc, char** argv)
{
    SampleINT8APIArgs args;
    bool argsOK = parseSampleINT8APIArgs(args, argc, argv);

    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (args.verbose)
    {
        sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);

    SampleINT8APIParams params;
    params = initializeSampleParams(args);
    // 创建本次测试代码最重要的对象
    SampleINT8API sample(params);
    sample::gLogInfo << "Building and running a INT8 GPU inference engine for " << params.modelFileName << std::endl;

    // build engine
    auto buildStatus = sample.build();
    if (buildStatus == sample::Logger::TestResult::kWAIVED)
    {
        return sample::gLogger.reportWaive(sampleTest);
    }
    else if (buildStatus == sample::Logger::TestResult::kFAILED)
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // 执行推理
    if (sample.infer() != sample::Logger::TestResult::kRUNNING)
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (sample.teardown() != sample::Logger::TestResult::kRUNNING)
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
