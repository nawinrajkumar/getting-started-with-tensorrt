#include <NvInfer.h>
#include <iostream>
#include <NvOnnxParser.h>

using namespace nvinfer1;


class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

IBuilder* builder = createInferBuilder(logger);


