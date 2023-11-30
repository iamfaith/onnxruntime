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
./build.sh --config=MinSizeRel --build_shared_lib --parallel --minimal_build --disable_ml_ops --disable_exceptions --disable_rtti --skip_tests --include_ops_by_config /home/faith/AI_baili_train/best5000-sim.required_operators.with_runtime_opt.config


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
