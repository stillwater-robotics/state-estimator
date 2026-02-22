cmake_minimum_required(VERSION 3.22)
project(StateEstimator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

set(DEPENDENCY_INCLUDES
  ../data-collector/include
  ../data-collector/third-party
)

set(SOURCES
    ./src/state_estimator.cpp
)

add_library(stateEstimator STATIC ${SOURCES})

target_include_directories(stateEstimator 
  PUBLIC 
    ./include
    ${DEPENDENCY_INCLUDES}
)

target_compile_features(stateEstimator PRIVATE cxx_std_17)

target_link_libraries(stateEstimator 
  PUBLIC 
    Eigen3::Eigen
)