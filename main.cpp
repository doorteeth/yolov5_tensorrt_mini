#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

class Loger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, char const *msg) noexcept override
    {
    }

} gLogger;

class PreprocessorTransform
{
public:
    PreprocessorTransform(const cv::Size &inputSize,
                          const double &f, const int &leftWidth,
                          const int &topHeight);

public:
    cv::Rect transformBbox(const cv::Rect &input) const;


private:
    cv::Size _inputSize;
    double _f;
    int _leftWidth;
    int _topHeight;
};

PreprocessorTransform::PreprocessorTransform(const cv::Size &inputSize, const double &f, const int &leftWidth,
                                             const int &topHeight):_inputSize(inputSize),_f(f),_leftWidth(leftWidth),_topHeight(topHeight)
{
}

cv::Rect PreprocessorTransform::transformBbox(const cv::Rect &input) const
{
    cv::Rect r;
    r.x = (input.x - _leftWidth) / _f;
    r.x = MAX(0, MIN(r.x, _inputSize.width - 1));

    r.y = (input.y - _topHeight) / _f;
    r.y = MAX(0, MIN(r.y, _inputSize.width - 1));

    r.width = input.width / _f;
    if (r.x + r.width > _inputSize.width)
    {
        r.width = _inputSize.width - r.x;
    }
    r.height = input.height / _f;
    if (r.y + r.height > _inputSize.height)
    {
        r.height = _inputSize.height - r.y;
    }
    return r;

}


class Detection
{
public:
    Detection(const int &classId, const cv::Rect &boundingBox, const double &score);

    const int32_t &classId() const noexcept;


    const cv::Rect &boundingBox() const noexcept;


    const double &score() const noexcept;


    const std::string &className() const noexcept;

private:
    int32_t _classId;
    std::string _className;

    cv::Rect _boundingBox;
    double _score;
};

Detection::Detection(const int &classId, const cv::Rect &boundingBox, const double &score)
        : _classId(classId), _boundingBox(boundingBox), _score(score)
{

}

const int32_t &Detection::classId() const noexcept
{
    return _classId;
}

const cv::Rect &Detection::boundingBox() const noexcept
{
    return _boundingBox;
}

const double &Detection::score() const noexcept
{
    return _score;
}

const std::string &Detection::className() const noexcept
{
    return _className;
}


void loadEngine(const std::string &filepath, std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                std::unique_ptr<nvinfer1::IExecutionContext> &context)
{
    std::ifstream file(filepath, std::ios::binary);
    std::vector<char> data;

    file.seekg(0, file.end);
    const auto size = file.tellg();
    file.seekg(0, file.beg);

    data.resize(size);
    file.read(data.data(), size);
    file.close();

    std::unique_ptr<nvinfer1::IRuntime> trtRuntime(nvinfer1::createInferRuntime(gLogger));
    engine.reset(trtRuntime->deserializeCudaEngine(data.data(), data.size()));
    context.reset(engine->createExecutionContext());

}

size_t getSizeDims(const nvinfer1::Dims &dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

void cudaGetMem(std::vector<nvinfer1::Dims> &input_dims, std::vector<nvinfer1::Dims> &output_dims,
                const std::unique_ptr<nvinfer1::ICudaEngine> &engine, std::vector<void *> &buffers)
{
    std::cout << buffers.size() << std::endl;

    std::cout << engine->getBindingDimensions(buffers.size() - 1).d[0] << "   "
              << engine->getBindingDimensions(buffers.size() - 1).d[1] << "   "
              << engine->getBindingDimensions(buffers.size() - 1).d[2] << std::endl;

    // CPU->GPU memory
    for (int i = 0; i < buffers.size(); ++i)
    {
        auto binding_size = getSizeDims(engine->getBindingDimensions(i)) * sizeof(float);

        cudaMalloc(&buffers[i], binding_size);

        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        } else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }

    }
//
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Failed load network" << std::endl;
        exit(1);
    }
}

void postprocessResults_0(float *gpu_output, const nvinfer1::Dims &dims,
                          const PreprocessorTransform &preprocessorTransform, std::vector<Detection> *out)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classes;

    const int nrClasses = dims.d[2] - 5;
    const int rowsize = dims.d[2];
    const int numGridBoxes = dims.d[1];

    std::vector<float> cpu_output(getSizeDims(dims));
    cudaMemcpyAsync(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float *begin = cpu_output.data(); //...[]............//...[]..........//

    for (int i = 0; i < numGridBoxes; ++i)
    {
        float *ptr = begin + i * rowsize;

        const float objectness = ptr[4];
        if (objectness < 0.4)
        {
            continue;
        }

        double maxClassScore = 0.0;
        int maxScoreIndex = 0;

        for (int j = 0; j < nrClasses; ++j)
        {
            const float &v = ptr[5 + j];
            if (v > maxClassScore)
            {
                maxClassScore = v;
                maxScoreIndex = j;
            }
        }
        const double score = objectness * maxClassScore;
        if (score < 0.4)
        {
            continue;
        }

        const float w = ptr[2];
        const float h = ptr[3];
        const float x = ptr[0] - w / 2.0;
        const float y = ptr[1] - h / 2.0;

        boxes.emplace_back(cv::Rect(x, y, w, h));
        scores.emplace_back(score);
        classes.emplace_back(maxScoreIndex);
    }

    std::vector<int> indices;

    cv::dnn::NMSBoxes(boxes, scores, 0.4, 0.4, indices);

    for (int i = 0; i < indices.size(); ++i)
    {
        const int &j = indices[i];
        const cv::Rect bbox = preprocessorTransform.transformBbox(boxes[j]);

        const double score = MAX(0.0, MIN(1., scores[j]));

        out->emplace_back(Detection(classes[j], bbox, score));
    }


}

void visualizeDetections(cv::Mat &image,
                         std::vector<Detection> &detections)
{
    for (unsigned int i = 0; i < detections.size(); ++i)
    {
        const Detection &det = detections[i];

        /*  bounding box  */
        const cv::Rect &bbox = det.boundingBox();

        std::cout<<bbox<<"  "<<std::endl;

        cv::rectangle(image, bbox, cv::Scalar(255, 0, 0), 2);

        /*  class  */
        std::string className = det.className();
        if (className.length() == 0)
        {
            const int classId = det.classId();
            className = std::to_string(classId);
        }
        cv::putText(image, className,
                    bbox.tl() + cv::Point(0, -10), cv::FONT_HERSHEY_PLAIN,
                    1.0, cv::Scalar(255, 255, 255));

        /*  score */
        const double score = det.score();
        cv::putText(image, std::to_string(score),
                    bbox.tl() + cv::Point(bbox.width, -10),
                    cv::FONT_HERSHEY_PLAIN, 1.0,
                    cv::Scalar(255, 255, 255));
    }
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
    std::string image_path = "../bus.jpg";
    std::string filepath="../test.engine";

    cv::Mat frame = cv::imread(image_path); //cpu

    // runtime:运行时候的接口实例
    //engine:序列化文件
    //context:管理中间激活的其他状态。

    std::unique_ptr<nvinfer1::IRuntime> trtRuntime(nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(nullptr);
    std::unique_ptr<nvinfer1::IExecutionContext> context(nullptr);

    loadEngine(filepath, engine, context);

    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void *> buffers(engine->getNbBindings());

    std::cout<<engine->getNbBindings()<<std::endl;

    cudaGetMem(input_dims, output_dims, engine, buffers);

    cv::cuda::GpuMat cuda_frame(frame);
    cv::cuda::Stream _cudaStream;

    const double f = MIN((double) input_dims[0].d[2] / frame.rows,
                         (double) input_dims[0].d[3] / frame.cols);

    const cv::Size boxSize = cv::Size(frame.cols * f, frame.rows * f);

//
    const int dr = input_dims[0].d[2] - boxSize.height;
    const int dc = input_dims[0].d[3] - boxSize.width;
    const int topHeight = std::floor(dr / 2.0);
    const int bottomHeight = std::ceil(dr / 2.0);
    const int leftWidth = std::ceil(dc / 2.0);
    const int rightWidth = std::floor(dc / 2.0);

    cv::cuda::resize(cuda_frame, cuda_frame, boxSize, 0, 0, cv::INTER_LINEAR, _cudaStream);
    cv::cuda::copyMakeBorder(cuda_frame, cuda_frame, topHeight, bottomHeight, leftWidth, rightWidth,
                             cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0), _cudaStream);

    cuda_frame.convertTo(cuda_frame, CV_32FC3, 1.0f / 255.0f, _cudaStream);

    int _networkRows = input_dims[0].d[2];
    int _networkCols = input_dims[0].d[3];
    std::cout << _networkRows << std::endl;
//
    const cv::Size networkSize(_networkCols, _networkRows);

    float *inputptr = (float *) buffers.at(0);
    std::vector<cv::cuda::GpuMat> channels;


    const int channelSize = networkSize.area();
    cudaMemcpy(inputptr,cuda_frame.data, 3*channelSize* sizeof(float),cudaMemcpyDeviceToDevice);

    //MAt [B] [G] [R]

    //MAT[0][0]  vec3d 11,13,25
    //                  |  |  |
    //MAT[0][1]  vec3d 88 99 100
    //  11 13  25   88 99 100
    //cudaMemcpy  11 13  25   88 99 100 ->buffers [       11 13  25   88 99 100                                                ]

    //11 88 ................  13 99 .................... 25 100...............
    //MAT

//    channels.push_back(cv::cuda::GpuMat(networkSize, CV_32FC1, inputptr + 2 * channelSize));
//    /*  G channel will go here  */
//    channels.push_back(cv::cuda::GpuMat(networkSize, CV_32FC1, inputptr + 1 * channelSize));
//    /*  R channel will go here  */
//    channels.push_back(cv::cuda::GpuMat(networkSize, CV_32FC1, inputptr));
//
//    cv::cuda::split(cuda_frame, channels, _cudaStream);



    context->enqueueV2(&buffers.front(), 0, nullptr);

    PreprocessorTransform preprocessorTransform(frame.size(), f, leftWidth, topHeight);//bbox recovery

    std::vector<Detection> lst;

    postprocessResults_0((float *) buffers.back(), output_dims.back(), preprocessorTransform, &lst);

    visualizeDetections(frame, lst);



//    cv::namedWindow("test");
//    cv::imshow("test", frame);
//    cv::waitKey(0);

    cv::imwrite("../result.jpg", frame);

    return 0;
}
