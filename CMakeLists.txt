project(Neun CXX)

export(PACKAGE Neun)

set(CMake_VERSION_MAJOR "0")
set(CMake_VERSION_MINOR "3")
set(CMake_VERSION_PATCH "2")

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS                "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-Wall -O0 -g -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG -std=c++11")

set(PROJECT_VERSION ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}.${CMake_VERSION_PATCH})

cmake_minimum_required(VERSION 2.8)

include(FindPkgConfig)

# Configure LibBoost
FIND_PACKAGE(Boost 1.45 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS} include)
link_directories(${Boost_LIBRARY_DIRS})

# Configure galib
set(GALIB_LIB ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/galib/ga/libga.a)
add_custom_target(galib COMMAND make lib
	          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/galib
			   )

add_subdirectory(include)
add_subdirectory(integrators)
add_subdirectory(archetypes)
add_subdirectory(concepts)
#	add_subdirectory(doc)
add_subdirectory(examples)
add_subdirectory(models)
add_subdirectory(wrappers)
add_subdirectory(optimizers)

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "An efficient library to simulate neural networks in C++.")
SET(CPACK_PACKAGE_VENDOR "Lareo and Herrero-Carrón")
set(CPACK_PACKAGE_CONTACT "angel.lareo@gmail.com")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Description")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright")
SET(CPACK_PACKAGE_VERSION_MAJOR ${CMake_VERSION_MAJOR})
SET(CPACK_PACKAGE_VERSION_MINOR ${CMake_VERSION_MINOR})
SET(CPACK_PACKAGE_VERSION_PATCH ${CMake_VERSION_PATCH})
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")

set(CPACK_DEBIAN_PACKAGE_SECTION "science")


set(CPACK_GENERATOR "DEB")

include(CPack)


configure_file(neun.pc.in neun.pc @ONLY)
install(FILES neun.pc DESTINATION pkgconfig)
