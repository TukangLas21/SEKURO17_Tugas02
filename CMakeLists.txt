cmake_minimum_required(VERSION 3.10)
project( Tugas02 )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)  

set(SOURCE_FILES main.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(Tugas02 main.cpp)
target_link_libraries( Tugas02 ${OpenCV_LIBS} )