find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# find_package(Pangolin REQUIRED)
# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

find_package(FMT REQUIRED)
find_package(OpenCV REQUIRED)
# 如果OpenCV被找到，包含目录
if(OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
else()
   message(FATAL_ERROR "OpenCV library not found")
endif()

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES} fmt)