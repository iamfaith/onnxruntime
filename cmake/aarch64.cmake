SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_C_COMPILER /home/faith/cross-pi-gcc-10.3.0-64/bin/aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /home/faith/cross-pi-gcc-10.3.0-64/bin/aarch64-linux-gnu-g++)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)


# SET(CMAKE_SYSTEM_NANE Android)
# SET(CMAKE_SYSTEM_PROCESSOR "armv7l")
# SET(ANDROID_ARCH_NAME "arm")
# SET(UNIX true)
# SET(CMAKE_C_COMPILER "gcc")
# SET(CMAKE_CXX_COMPILER "g++")


# -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64.cmake

# git clone --recursive https://github.com/Microsoft/onnxruntime -b v1.15.1 onnxruntime-v1.15.1
# cd onnxruntime-v1.15.1
# mkdir build
# cd build
# cmake -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=../../aarch64.cmake -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_DEV_MODE=OFF -DONNX_CUSTOM_PROTOC_EXECUTABLE=/home/cyrus/work/tmp/protoc-3.21.12/bin/protoc ../cmake/
# make -j $(nproc)



# 1. https://onnxruntime.ai/docs/build/inferencing.html#cross-compiling-on-linux
# 2. https://www.linaro.org/downloads/#gnu_and_llvm
# 3. https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads


# https://releases.linaro.org/components/toolchain/binaries
