// #include "include/onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <assert.h>
#include <sys/time.h>
// #include <utility>
#include "utils.h"
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <iostream>
#else
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

int cvRound(double value) { return (ceil(value)); }
#endif

static inline uint64_t time_get(void) {
  struct timeval tv;

  gettimeofday(&tv, 0);
  return (uint64_t)(tv.tv_sec * 1000000000ULL + tv.tv_usec * 1000);
}

// 1. sudo ldconfig -v|grep onnx   check so
// 2. export LD_LIBRARY_PATH=`pwd`
// 3. gcc file1.o file2.o /usr/lib/libonnxruntime.so.1.17.0 -o myapp3

// g++ -O3 test.cc -L`pwd` -lonnxruntime && ./a.out
// sudo ldconfig -p | grep onnxsh
// https://zhuanlan.zhihu.com/p/513777076

void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                      float& bestConf, int& bestClassId) {
  // first 5 element are box and obj confidence
  bestClassId = 5;
  bestConf = 0;

  for (int i = 5; i < numClasses + 5; i++) {
    if (it[i] > bestConf) {
      bestConf = it[i];
      bestClassId = i - 5;
    }
  }
}

std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                      const cv::Size& originalImageShape,
                                      std::vector<Ort::Value>& outputTensors,
                                      const float& confThreshold, const float& iouThreshold) {
  std::vector<cv::Rect> boxes;
  std::vector<float> confs;
  std::vector<int> classIds;

  auto* rawOutput = outputTensors[0].GetTensorData<float>();
  std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<float> output(rawOutput, rawOutput + count);

  // for (const int64_t& shape : outputShape)
  //     std::cout << "Output Shape: " << shape << std::endl;

  // first 5 elements are box[4] and obj confidence
  int numClasses = (int)outputShape[2] - 5;
  int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

  // only for batch size = 1
  for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2]) {
    float clsConf = it[4];

    if (clsConf > confThreshold) {
      int centerX = (int)(it[0]);
      int centerY = (int)(it[1]);
      int width = (int)(it[2]);
      int height = (int)(it[3]);
      int left = centerX - width / 2;
      int top = centerY - height / 2;

      float objConf;
      int classId;
      /////////
      getBestClassInfo(it, numClasses, objConf, classId);

      float confidence = clsConf * objConf;

      boxes.emplace_back(left, top, width, height);
      confs.emplace_back(confidence);
      classIds.emplace_back(classId);
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
  // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

  std::vector<Detection> detections;

  for (int idx : indices) {
    Detection det;
    det.box = cv::Rect(boxes[idx]);
    utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

    det.conf = confs[idx];
    det.classId = classIds[idx];
    detections.emplace_back(det);
  }

  return detections;
}

int main(void) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  std::cout << "Environment created" << std::endl;

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(4);

  // session_options.SetIntraOpNumThreads(std::min(6, (int) std::thread::hardware_concurrency()));
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "best5000-sim.with_runtime_opt.ort";
#endif

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    // char* input_name = session.GetInputNameAllocated(i, allocator).get();
    // printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = "images";

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 1 * 3 * 384 * 768;  // simplify ... using known dim values to calculate size
                                                 // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"output"};

  ////////////////////////////////////// fake data
  // initialize input data with values in [0.0, 1.0]
  // for (unsigned int i = 0; i < input_tensor_size; i++)
  //   input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // // create input tensor object from data values
  // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  // assert(input_tensor.IsTensor());
  //////////////////////////////////////////fake data

  char* imagepath = "/home/faith/AI_baili_train/images/22.png";
  cv::Mat image = cv::imread(imagepath, 1);
  cv::Mat resizedImage, m;
#if (CV_VERSION_MAJOR >= 4)
  cv::cvtColor(image, m, cv::COLOR_BGR2RGB);
#else
  cv::cvtColor(image, m, CV_BGR2RGB);
#endif

  cv::Mat floatImage;
  cv::Size inputImageShape = cv::Size(768, 768);
  bool isDynamicInputShape = false;

  utils::letterbox(m, resizedImage, cv::Size2f(inputImageShape),
                   cv::Scalar(114, 114, 114), true, false, true, 32);

  // inputTensorShape[2] = resizedImage.rows;
  // inputTensorShape[3] = resizedImage.cols;

  resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
  float* blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
  cv::Size floatImageSize{floatImage.cols, floatImage.rows};

  // hwc -> chw
  std::vector<cv::Mat> chw(floatImage.channels());
  for (int i = 0; i < floatImage.channels(); ++i) {
    chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
  }
  cv::split(floatImage, chw);

  ///  finish preprocess
  size_t inputTensorSize = 1 * 3 * 384 * 768;
  std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

  std::vector<Ort::Value> input_tensor;

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  std::vector<int64_t> inputTensorShape{1, 3, 384, 768};
  input_tensor.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size()));

  // for (int i = 0; i < 30; i++) {
  //   int offset = 100 * 768 + 500;
  //   std::cout << inputTensorValues[i + offset] << " ";
  // }

  int times = 10;
  int loop = times;
  double avg = 0;
  while (loop > 0) {
    uint64_t begin = time_get();

    // score model & input tensor, get back output tensor
    // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensor.data(), 1, output_node_names.data(), 1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    const float confThreshold = 0.55;
    const float iouThreshold = 0.45;
    std::vector<Detection> detections = postprocessing(resizedShape,
                                     image.size(),
                                     output_tensors,
                                     confThreshold, iouThreshold);
    std::cout << detections.size() << std::endl;
    for (const Detection& detection : detections) {
      std::cout << detection << std::endl;
    }
    //   assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    uint64_t end = time_get();
    double lasting = (double)(end - begin) / 1000.0f / 1000.0f;
    printf("\nLasting: %12.3fms\n", lasting);

    // ///******************  comment out
    // // Get pointer to output tensor float values
    // float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // //   assert(abs(floatarr[0] - 0.000045) < 1e-6);

    // // score the model, and print scores for first 5 classes
    // for (int i = 0; i < 9; i++)
    //   printf("Score for class [%d] =  %f\n", i, floatarr[i]);

    // // 获取output的shape
    // Ort::TensorTypeAndShapeInfo shape_info = output_tensors.front().GetTensorTypeAndShapeInfo();

    // // 获取output的dim
    // size_t dim_count = shape_info.GetDimensionsCount();
    // std::cout<< dim_count << std::endl;

    // // 获取output的shape
    // int64_t dims[3];
    // shape_info.GetDimensions(dims, sizeof(dims) / sizeof(dims[0]));
    // std::cout<< dims[0] << "," << dims[1] << "," << dims[2] << std::endl;

    avg += lasting;
    loop--;
  }
  printf("avg: %12.3fms\n", avg / times);

  printf("Done!\n");

  delete[] blob;

  //   // 取output数据
  //   float* f = output_tensors.GetTensorMutableData<float>();
  //   for (int i = 0; i < dims[1]; i++) {
  //     std::cout << f[i] << std::endl;
  //   }

  return 0;
}
