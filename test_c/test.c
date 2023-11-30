#include "include/onnxruntime_c_api.h"
// #include "include/onnxruntime_cxx_api.h"
// gcc test.c -L`pwd` -lonnxruntime
// g++ test.c -L`pwd` -lonnxruntime
// sudo ldconfig -p | grep onnxsh
// https://zhuanlan.zhihu.com/p/513777076
int main(void) {
    //模型初始化参数
    char* model_path    = "/home/faith/AI_baili_train/best5000-sim.onnx";
    const char* inputNames[]  = {"xxx"}; //输入节点名
    const char* outputNames[] = {"xxx"}; //输出节点名
    int inputNodeNum = 1;//输入节点个数
    int outputNodeNum = 1; //输出节点个数

    int input_w = 760;//模型输入width
    int input_h = 320;//模型输入height

    //H*W*C
    size_t model_input_ele_count = input_h * input_w * 3;

    //input data
    float* model_input = (float*)malloc(sizeof(float) * model_input_ele_count);

    //{N,C,H,W}
    int64_t input_shape[4];
    input_shape[0] = 1;
    input_shape[1] = 3;
    input_shape[2] = input_h;
    input_shape[3] = input_w;

    size_t input_shape_len = 4;//dim = 4

    size_t model_input_len = model_input_ele_count * sizeof(float);

    //初始化OrtApi
    const OrtApi* g_ort = NULL;
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    //初始化OrtEnv
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

    //初始化session_options
    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);


    //初始化session
    OrtSession* session;
    g_ort->CreateSession(env, model_path, session_options, &session);

   

    return 0;
}
