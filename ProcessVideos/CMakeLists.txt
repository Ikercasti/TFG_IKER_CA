project( TrackAprox )
cmake_minimum_required(VERSION 3.25)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/RaspberryCam/Include/)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/RaspberryCam/Source/)
set(HEADERFILES ${INCLUDE_DIR}OCV_Funcs.hpp ${INCLUDE_DIR}spline.h)
set(SOURCEFILES ${SOURCE_DIR}OCV_Funcs.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
message(OpenCV_DIR="${OpenCV_DIR}")

include_directories( ${PROJECT_DIR} )

add_executable( ${PROJECT_NAME} TrackAprox.cpp ${HEADERFILES} ${SOURCEFILES} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

#add_executable( original TrackAprox_orig.cpp ${HEADERFILES} ${SOURCEFILES} )
#target_link_libraries( original ${OpenCV_LIBS} )    