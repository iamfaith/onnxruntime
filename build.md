如果您想要修改为CPU运行，裁剪算子，您可以参考以下步骤：

- 首先，您需要下载onnxruntime的源码包，并解压到一个目录下³。
- 然后，您可以在该目录下运行build.sh脚本，指定您想要使用的算子和执行器，例如：
  - `./build.sh --skip_tests --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config onnxruntime/test/testdata/required_ops.config` 表示裁剪算子，只保留required_operators.config文件中指定的算子³。
  - `./build.sh --use_cpu` 表示使用CPU执行器⁴。
- 最后，您可以在build目录下找到编译好的库文件和头文件，然后在您的项目中链接和使用它们²。


contrib_ops: bert等可以disable

./build.sh --skip_tests --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config onnxruntime/test/testdata/required_ops.config --parallel 16



python tools/python/create_reduced_build_config.py /home/faith/AI_baili_train/best5000-sim.onnx
python tools/python/convert_onnx_models_to_ort.py /home/faith/AI_baili_train/best5000-sim.onnx


./build.sh --skip_tests --config Release --build_shared_lib --parallel 16 --include_ops_by_config /home/faith/AI_baili_train/required_operators.config --minimal_build --disable_ml_ops --disable_exceptions



https://blog.csdn.net/qq_41182465/article/details/130901506

--- mini
./build.sh --config=MinSizeRel --build_shared_lib --parallel 16 --minimal_build --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config /home/faith/AI_baili_train/best5000-sim.required_operators.with_runtime_opt.config

cross compile:
sudo apt-get install --reinstall gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
cd build/Linux/MinSizeRel
/home/faith/miniconda3/bin/cmake /home/faith/onnxruntime/cmake -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -DPython_EXECUTABLE=/home/faith/miniconda3/bin/python3 -DPYTHON_EXECUTABLE=/home/faith/miniconda3/bin/python3 -Donnxruntime_USE_MIMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_BUILD_JAVA=OFF -Donnxruntime_BUILD_NODEJS=OFF -Donnxruntime_BUILD_OBJC=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_BUILD_APPLE_FRAMEWORK=OFF -Donnxruntime_USE_DNNL=OFF -Donnxruntime_USE_NNAPI_BUILTIN=OFF -Donnxruntime_USE_RKNPU=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_VITISAI=OFF -Donnxruntime_USE_TENSORRT=OFF -Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_TVM_CUDA_RUNTIME=OFF -Donnxruntime_TVM_USE_HASH=OFF -Donnxruntime_USE_MIGRAPHX=OFF -Donnxruntime_DISABLE_CONTRIB_OPS=OFF -Donnxruntime_DISABLE_ML_OPS=ON -Donnxruntime_DISABLE_RTTI=ON -Donnxruntime_DISABLE_EXCEPTIONS=ON -Donnxruntime_MINIMAL_BUILD=ON -Donnxruntime_EXTENDED_MINIMAL_BUILD=OFF -Donnxruntime_MINIMAL_BUILD_CUSTOM_OPS=OFF -Donnxruntime_REDUCED_OPS_BUILD=ON -Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=OFF -Donnxruntime_USE_DML=OFF -Donnxruntime_USE_WINML=OFF -Donnxruntime_BUILD_MS_EXPERIMENTAL_OPS=OFF -Donnxruntime_USE_TELEMETRY=OFF -Donnxruntime_ENABLE_LTO=OFF -Donnxruntime_USE_ACL=OFF -Donnxruntime_USE_ACL_1902=OFF -Donnxruntime_USE_ACL_1905=OFF -Donnxruntime_USE_ACL_1908=OFF -Donnxruntime_USE_ACL_2002=OFF -Donnxruntime_USE_ARMNN=OFF -Donnxruntime_ARMNN_RELU_USE_CPU=ON -Donnxruntime_ARMNN_BN_USE_CPU=ON -Donnxruntime_USE_JSEP=OFF -Donnxruntime_ENABLE_NVTX_PROFILE=OFF -Donnxruntime_ENABLE_TRAINING=OFF -Donnxruntime_ENABLE_TRAINING_OPS=OFF -Donnxruntime_ENABLE_TRAINING_APIS=OFF -Donnxruntime_ENABLE_CPU_FP16_OPS=OFF -Donnxruntime_USE_NCCL=OFF -Donnxruntime_BUILD_BENCHMARKS=OFF -Donnxruntime_USE_ROCM=OFF -DOnnxruntime_GCOV_COVERAGE=OFF -Donnxruntime_USE_MPI=OFF -Donnxruntime_ENABLE_MEMORY_PROFILE=OFF -Donnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO=OFF -Donnxruntime_USE_CUDA_NHWC_OPS=OFF -Donnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=OFF -Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING=ON -Donnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING=OFF -Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING=ON -Donnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER=OFF -Donnxruntime_ENABLE_WEBASSEMBLY_THREADS=OFF -Donnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO=OFF -Donnxruntime_ENABLE_WEBASSEMBLY_PROFILING=OFF -Donnxruntime_ENABLE_LAZY_TENSOR=OFF -Donnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS=OFF -Donnxruntime_ENABLE_CUDA_PROFILING=OFF -Donnxruntime_ENABLE_ROCM_PROFILING=OFF -Donnxruntime_USE_XNNPACK=OFF -Donnxruntime_USE_WEBNN=OFF -Donnxruntime_USE_CANN=OFF -Donnxruntime_USE_TRITON_KERNEL=OFF -Donnxruntime_DISABLE_FLOAT8_TYPES=OFF -Donnxruntime_DISABLE_SPARSE_TENSORS=OFF -Donnxruntime_DISABLE_OPTIONAL_TYPE=OFF -DCMAKE_TLS_VERIFY=ON -DFETCHCONTENT_QUIET=OFF -Donnxruntime_ENABLE_MEMLEAK_CHECKER=OFF -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_PREFIX_PATH=/home/faith/onnxruntime/build/Linux/MinSizeRel/installed -DCMAKE_TOOLCHAIN_FILE=/home/faith/onnxruntime/cmake/aarch64.cmake


workable:
download: https://sourceforge.net/projects/raspberry-pi-cross-compilers/
disable onnxruntime_ENABLE_CPU_FP16_OPS
1. CROSS_COMPILE=/home/faith/cross-pi-gcc-10.3.0-64/bin/aarch64-linux-gnu-g++
2. mkdir arm_build
3. cmake -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=OFF -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=/home/faith/onnxruntime/cmake/aarch64.cmake -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_DEV_MODE=OFF -DCMAKE_PREFIX_PATH=/home/faith/onnxruntime/arm_build/installed ../cmake/

native raspberry:
/home/faith/.local/bin/cmake -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=OFF -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Re
lease -Dprotobuf_WITH_ZLIB=OFF  -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_DEV_MODE=OFF ../cmake/

4. make -j $(nproc)

or

5. not good
./build.sh --config=MinSizeRel --build_shared_lib --parallel 16 --minimal_build --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config /home/faith/AI_baili_train/best5000-sim.required_operators.with_runtime_opt.config --build_dir arm_build --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=/home/faith/onnxruntime/cmake/aarch64.cmake


remove minimal_build:（python tools/python/create_reduced_build_config.py /home/faith/AI_baili_train/best5000-sim.onnx）Best worable!!!!!!
./build.sh --config=Release --build_shared_lib --parallel 16 --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config /home/faith/AI_baili_train/best5000-sim.required_operators.config --build_dir arm_build --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=/home/faith/onnxruntime/cmake/aarch64.cmake



重新build的话需要： --update --build
./build.sh --arm --config=MinSizeRel --build_shared_lib --parallel 16 --minimal_build --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config /home/faith/AI_baili_train/best5000-sim.required_operators.with_runtime_opt.config --update --build --build_dir build/x86


# cross compile
https://github.com/PINTO0309/onnxruntime4raspberrypi

1. docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
2. docker run --rm -it \
-v ${PWD}:/workdir \
pinto0309/raspios_lite_armhf:2021-03-04_buster \
/bin/bash


https://github.com/lukechilds/dockerpi/issues/6

sudo systemctl enable ssh
sudo systemctl start ssh

docker run --rm -it \
-v ${PWD}:/sdcard/boot/onnxruntime \
-p 5022:22 \
lukechilds/dockerpi


3. apt install -y protobuf-compiler libcurl4-openssl-dev \
libatlas-base-dev git wget make python3-pip cmake

python3 -m pip install flatbuffers
python3 -m pip install -r requirements-dev.txt

fatal: detected dubious ownership in repository at '/home/faith/onnxruntime':
git config --global --add safe.directory '*'

CMake 3.26 or higher is required.  You are running version 3.16.3:
Enable the arm64 architecture (this matches aarch64):
python3 -m pip install cmake

4. ./build.sh --arm --config=MinSizeRel --build_shared_lib --parallel 16 --minimal_build --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config best5000-sim.required_operators.with_runtime_opt.config --allow_running_as_root --update --build

native:
./build.sh --config=MinSizeRel --build_shared_lib --parallel 16 --minimal_build --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config best5000-sim.required_operators.with_runtime_opt.config --allow_running_as_root


--skip_submodule_sync  no sync


build for arm:
https://github.com/microsoft/onnxruntime/issues/2684
To Reproduce

sudo apt-get update
sudo apt-get install -y \
    sudo \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    tar
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --upgrade wheel
pip3 install numpy
cd /code
git clone --recursive https://github.com/Microsoft/onnxruntime

cd /code/onnxruntime
./build.sh --config MinSizeRel --arm --update --build




docker run --rm -it -v $PWD:/home/faith/onnxruntime cross-stretch bash


根据我从网络上搜索到的信息，编译的时候可以使用以下命令来引用libonnxruntime.so文件：

- 如果您使用的是gcc或g++，您可以使用`-L`选项来指定动态链接库的路径，例如`-L/usr/local/lib`，然后使用`-l`选项来指定动态链接库的名称，例如`-lonnxruntime`¹³。
- 如果您使用的是cmake，您可以使用`find_library`函数来查找动态链接库的位置，然后使用`target_link_libraries`函数来链接动态链接库，例如：

```cmake
find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib)
target_link_libraries(your_project ${ONNXRUNTIME_LIB})
```

⁴

我希望这些信息对您有所帮助。如果您有其他问题，欢迎继续与我交流。😊

Source: Conversation with Bing, 11/30/2023
(1) onnxruntime (C++/CUDA) 编译安装及部署 - CSDN博客. https://blog.csdn.net/weixin_44684139/article/details/123504222.
(2) C/C++下的ONNXRUNTIME推理 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/513777076.
(3) 配置onnxruntime C++部署环境 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/635363825.
(4) Install ONNX Runtime | onnxruntime. https://onnxruntime.ai/docs/install/.
(5) undefined. https://github.com/Microsoft/onnxruntime.



您好，感谢您的提问。😊

根据我的搜索结果，我发现有一些方法可以在编译onnxruntime时跳过contrib_ops的编译，从而减少编译时间和生成的库文件的大小。这些方法主要有以下两种：

- 使用create_reduced_build_config.py脚本来根据ONNX文件生成所需的算子配置文件，然后在build.sh命令中使用--include_ops_by_config参数来指定该配置文件³。例如，如果您的ONNX文件名是model.onnx，您可以执行以下命令：

```bash
python create_reduced_build_config.py ./model.onnx
./build.sh --skip_tests --config Release --build_shared_lib --parallel --include_ops_by_config ./required_operators.config
```

- 使用--minimal_build参数来编译onnxruntime，这会只包含最基本的算子和功能，然后在运行时使用ORT format模型来加载所需的算子⁶。ORT format模型是一种优化后的二进制格式，可以通过onnxruntime_tools.converter.convert_onnx_model_to_ort函数来从ONNX模型转换得到。例如，如果您的ONNX文件名是model.onnx，您可以执行以下命令：

```bash
./build.sh --skip_tests --config Release --build_shared_lib --parallel --minimal_build
python -m onnxruntime_tools.converter.convert_onnx_model_to_ort model.onnx
```

希望这些方法对您有帮助。如果您还有其他问题，欢迎继续提问。😊

Source: Conversation with Bing, 11/30/2023
(1) OnnxRuntime编译_onnxruntime 编译_越长大越孤单wz的博客-CSDN博客. https://blog.csdn.net/qq_41182465/article/details/130901506.
(2) undefined. https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md.
(3) Contrib Ops · microsoft/onnxruntime Wiki · GitHub. https://github.com/microsoft/onnxruntime/wiki/Contrib-Ops.
(4) Contrib operators | onnxruntime. https://onnxruntime.ai/docs/reference/operators/ContribOperators.html.
(5) ubuntu下的onnxruntime(c++)编译_点py的博客-爱代码爱编程. https://icode.best/i/99456251756343.
(6) Ubuntu下的onnxruntime(c++)编译_51CTO博客_ubuntu编译c程序. https://blog.51cto.com/u_15699099/5649211.
(7) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.15.0/docs/ContribOperators.md.
(8) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ContribOperators.md.
(9) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.13.1/docs/ContribOperators.md.
(10) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.12.0/docs/ContribOperators.md.
(11) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.11.0/docs/ContribOperators.md.
(12) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.10.0/docs/ContribOperators.md.
(13) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/ContribOperators.md.
(14) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.8.0/docs/ContribOperators.md.
(15) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.7.0/docs/ContribOperators.md.







我希望这些信息对您有所帮助。如果您还有其他问题，欢迎继续与我交流。😊

Source: Conversation with Bing, 2023/11/30
(1) [推理部署] 全网最详细 ONNXRuntime C++/Java/Python 资料！ - 知乎. https://zhuanlan.zhihu.com/p/414317269.
(2) 在ONNX Runtime框架下实现YOLOv8部署：同时支持目标检测和实例分割；还支持自定义训练结果 - 知乎. https://zhuanlan.zhihu.com/p/648704225.
(3) ONNX模型推理使用多核CPU加速推理 - CSDN博客. https://blog.csdn.net/weixin_43999691/article/details/130720771.
(4) ONNXRuntime整体概览 - 知乎. https://zhuanlan.zhihu.com/p/346544539.

disable ml ops是一个编译选项，它的意思是禁用机器学习相关的算子，比如LSTM，GRU，RNN等³。这样可以减少编译后的二进制文件的大小，如果您的模型不需要这些算子的话。您可以在build.sh脚本中添加--disable_ml_ops参数来启用这个选项¹。

Source: Conversation with Bing, 2023/11/30
(1) MinGW support (MSYS2) · Issue #10976 · microsoft/onnxruntime - GitHub. https://github.com/microsoft/onnxruntime/issues/10976.
(2) Custom build | onnxruntime. https://onnxruntime.ai/docs/build/custom.html.
(3) Contrib operators | onnxruntime. https://onnxruntime.ai/docs/reference/operators/ContribOperators.html.
(4) undefined. https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md.
(5) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.15.0/docs/ContribOperators.md.
(6) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ContribOperators.md.
(7) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.13.1/docs/ContribOperators.md.
(8) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.12.0/docs/ContribOperators.md.
(9) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.11.0/docs/ContribOperators.md.
(10) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.10.0/docs/ContribOperators.md.
(11) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/ContribOperators.md.
(12) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.8.0/docs/ContribOperators.md.
(13) undefined. https://github.com/microsoft/onnxruntime/blob/rel-1.7.0/docs/ContribOperators.md.







# contrib_ops

您好，这是Bing。我可以帮您了解onnxruntime_DISABLE_CONTRIB_OPS这个参数的作用和contrib_ops包含了什么。😊

根据我从网络上搜索到的信息，onnxruntime_DISABLE_CONTRIB_OPS这个参数是用来控制是否编译contrib_ops域中的运算符的。¹²

contrib_ops域中包含了一些默认内置在运行时中的运算符，它们是为了满足一些特定的需求而添加的，但是并不是标准的ONNX运算符。¹²

只有一些被选中的运算符才会被添加到contrib_ops域中，以避免增加核心运行时包的二进制大小。¹²

如果可能的话，应该使用自定义运算符来实现特殊的功能。¹²

contrib_ops域中包含了以下运算符：³

- Attention
- BiasGelu
- BiasSoftmax
- ConvTransposeWithDynamicPads
- DynamicQuantizeLSTM
- EmbedLayerNormalization
- FastGelu
- GatherND
- Gelu
- LayerNormalization
- LongformerAttention
- MatMulInteger16
- MurmurHash3
- QLinearConv
- QLinearMatMul
- QLinearSigmoid
- QLinearLeakyRelu
- QLinearAdd
- QLinearMul
- QLinearAveragePool
- QLinearGlobalAveragePool
- QLinearReduceMean
- QLinearActivation
- QAttention
- QuantizeLinear
- Range
- SkipLayerNormalization
- SoftmaxCrossEntropyLoss
- SparseToDense
- StringNormalizer
- Tokenizer
- TransposeMatMul
- Unique
- WordConvEmbedding

如果您想编译onnxruntime时不包含contrib_ops域中的运算符，您需要在CMakeLists.txt文件中添加一行set(onnxruntime_DISABLE_CONTRIB_OPS ON)。¹²

希望这些信息能够帮助您理解onnxruntime_DISABLE_CONTRIB_OPS这个参数的作用和contrib_ops包含了什么。如果您还有其他问题，欢迎继续与我交流。😊

Source: Conversation with Bing, 12/1/2023
(1) Contrib operators | onnxruntime. https://onnxruntime.ai/docs/reference/operators/ContribOperators.html.
(2) Contrib Ops · microsoft/onnxruntime Wiki · GitHub. https://github.com/microsoft/onnxruntime/wiki/Contrib-Ops.
(3) Contrib Operator Schemas - GitHub: Let’s build from here. https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md.
