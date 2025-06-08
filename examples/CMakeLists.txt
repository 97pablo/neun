set (INCLUDE_DIR ../include)
include_directories(${INCLUDE_DIR} ../concepts ../models ../integrators ../wrappers ../archetypes ../optimizers ../thirdparty/galib)

add_executable(basic basic.cpp)
target_link_libraries(basic)

add_executable(synapsis synapsis.cpp)
target_link_libraries(synapsis)

add_executable(network network.cpp)
target_link_libraries(network)

add_executable(basicGA basicGA.cpp)
target_link_libraries(basicGA PRIVATE ${GALIB_LIB})

add_executable(networkGA networkGA.cpp)
target_link_libraries(networkGA PRIVATE ${GALIB_LIB})