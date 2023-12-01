# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (test LANGUAGES CXX)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set_source_files_properties(${PROJECT_NAME}.cc PROPERTIES LANGUAGE CXX )
# set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}) #code as library

aux_source_directory(src src)
message(STATUS "src:  ${src}")
# Add an executable
add_executable(${PROJECT_NAME} ${src})
# add_executable(${PROJECT_NAME}
#                             src/${PROJECT_NAME}.cc
#                             src/utils.cc)

find_package(OpenCV QUIET COMPONENTS opencv_world)
# for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
if("${OpenCV_LIBS}" STREQUAL "")
    set(OpenCV_FOUND FALSE)
endif()
if(NOT OpenCV_FOUND)
    message(STATUS "finding OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio")
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
endif()
if(NOT OpenCV_FOUND)
    message(STATUS "finding OpenCV QUIET COMPONENTS core highgui imgproc")
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()


message(STATUS "OpenCV_FOUND = ${OpenCV_FOUND}")

# set(OpenCV_FOUND FALSE)

set(ONNXRUNTIME_ROOT /home/faith/onnxruntime/test_c)
include_directories(${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime ${ONNXRUNTIME_ROOT}/lib)
# link_directories(${ONNXRUNTIME_ROOT}/lib)

message(STATUS "ONNXRUNTIME_ROOT = ${ONNXRUNTIME_ROOT}")


if(OpenCV_FOUND)
    # need to set independently
 #   set(OpenCV_INSTALL_PATH /opt/opencv-master)
    # set(OpenCV_INSTALL_PATH /home/pi/opencv-master)

    message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

    if(${OpenCV_VERSION_MAJOR} GREATER 3)
        set(CMAKE_CXX_STANDARD 11)
    endif()

    # target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

    # target_link_options(${PROJECT_NAME} PRIVATE -static)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY} opencv_dnn)
else()
    # target_compile_definitions(${name} PUBLIC USE_NCNN_SIMPLEOCV)
    # target_link_libraries(${name} PRIVATE ncnn)
    # target_link_libraries( ${PROJECT_NAME}
    #     PRIVATE
    #     monnx m ocv
    # )
endif()


install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin)


# Include the CPack module
include(CPack)

# Set the CPack generator to ZIP
set(CPACK_GENERATOR ZIP)

# Set some CPack variables for the package metadata
set(CPACK_PACKAGE_NAME "Yolo")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION "A simple C++ application with a dynamic library dependency")




# cmake_minimum_required(VERSION 2.8.9)
# project (Logging)
# include_directories(include)
# file(GLOB LOGGINGSOURCES "libsrc/*.cpp")
# file(GLOB SOURCES "src/*.cpp")
# add_library(convertString SHARED ${LOGGINGSOURCES})
# install(TARGETS convertString DESTINATION /root/Deepak/)
# link_directories( /root/Deepak/ )
# set(PROJECT_LINK_LIBS libconvertString.so)
# add_executable(hello ${SOURCES})
# target_link_libraries(hello ${PROJECT_LINK_LIBS} )
