/**
    @brief A demo of inference by loading  deserialize yolov5.engine
    @author Jinfu Liu
    @date 2023.09.15
    @version 1.0  
*/

#include "cuda_utils.h"
#include "logging.h"
#include "preprocess.h"
#include "postprocess.hpp"

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>

using namespace nvinfer1;
static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YoloV5_Infer{
private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    std::string engine_name = "../yolov5s.engine";
    cudaStream_t stream;
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;

public:
    YoloV5_Infer(){};

    cv::Mat run(cv::Mat& img){
        // deserialize
        deserialize_engine(engine_name, &runtime, &engine, &context);
        CUDA_CHECK(cudaStreamCreate(&stream));
        prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);
        
        // preprocess
        cuda_preprocess_init(kMaxInputImageSize);
        cuda_preprocess(img.ptr(), img.cols, img.rows, &gpu_buffers[0][0], kInputW, kInputH, stream);

        // Run inference
        auto start = std::chrono::system_clock::now();
        infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
        auto end = std::chrono::system_clock::now();

        // Postprocess
        std::vector<Detection> res;
        nms(res, &cpu_output_buffer[0], kConfThresh, kNmsThresh); // NMS
        draw_bbox(img, res); // Draw bounding boxes
        return img;
    }

    // deserialize
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            assert(false);
        }
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char* serialized_engine = new char[size];
        assert(serialized_engine);
        file.read(serialized_engine, size);
        file.close();

        *runtime = createInferRuntime(gLogger);
        assert(*runtime);
        *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
        assert(*engine);
        *context = (*engine)->createExecutionContext();
        assert(*context);
        delete[] serialized_engine;
    }

    // prepare buffers
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
        assert(engine->getNbBindings() == 2);
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));
        *cpu_output_buffer = new float[kBatchSize * kOutputSize];
    }

    // inference
    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
        context.enqueue(batchsize, gpu_buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }
};