find_package(Pangolin REQUIRED)

find_package(OpenCV REQUIRED)
# 如果OpenCV被找到，包含目录
if(OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
else()
   message(FATAL_ERROR "OpenCV library not found")
endif()

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})