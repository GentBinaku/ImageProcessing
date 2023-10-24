cmake_minimum_required(VERSION 3.25)
project(ImageProcessing)

function(copy_openv_dlls target_name)
    if (MSVC)
        file(GLOB OpenCV_DLLS ${PROJECT_SOURCE_DIR}/external/opencv/bin/Debug/*.dll)
        foreach (OpenCV_DLL ${OpenCV_DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${OpenCV_DLL} $<TARGET_FILE_DIR:${target_name}>)
        endforeach ()
    endif ()
endfunction()

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED PATHS ${PROJECT_SOURCE_DIR}/external/opencv)
# TODO: Use `Development.Embed` component when requiring cmake >= 3.18
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)


include(GNUInstallDirs)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})


add_executable(ImageProcessing main.cpp)

target_link_libraries(ImageProcessing INTERFACE
        Python3::Python
        Python3::Module
)

if(Python3_NumPy_FOUND)
    target_link_libraries(ImageProcessing INTERFACE
            Python3::NumPy
    )
else()
    target_compile_definitions(ImageProcessing INTERFACE WITHOUT_NUMPY)
endif()

target_include_directories( ImageProcessing  PRIVATE ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${Python_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(ImageProcessing PRIVATE ${OpenCV_LIBS} ${Python3_LIBRARIES} )

copy_openv_dlls(ImageProcessing)

