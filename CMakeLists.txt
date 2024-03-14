cmake_minimum_required(VERSION 3.15)
project(cmax_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "-fopenmp -O4 ${CMAKE_CXX_FLAGS}")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(gsl REQUIRED gsl)

find_package(OpenCV REQUIRED)

file(GLOB SOURCES src/frontend/*.cpp
                  src/backend/*.cpp
                  src/utils/*.cpp
                  src/*.cpp)

cs_add_executable(${PROJECT_NAME} ${SOURCES})

include_directories(include)
include_directories(thirdparty/basalt-headers/thirdparty/Sophus)
include_directories(thirdparty/basalt-headers/thirdparty/cereal/include)
include_directories(thirdparty/basalt-headers/include)

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${gsl_LIBRARIES}
)

cs_export()
