cmake_minimum_required(VERSION 3.0)
project(imageBasics)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 如果OpenCV被找到，包含目录
if(OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
else()
   message(FATAL_ERROR "OpenCV library not found")
endif()

# 添加可执行文件
add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})
