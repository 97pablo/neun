
add_library(optimizers INTERFACE)

target_include_directories(optimizers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/thirdparty/galib
)

link_libraries("-static")
add_dependencies(optimizers galib)
set_target_properties(optimizers PROPERTIES LINK_FLAGS "-static" )
target_link_libraries(optimizers INTERFACE ${CMAKE_SOURCE_DIR}/thirdparty/galib/ga/libga.a)


install(FILES
    NeuronGAOptimizer.h
    VoltageDifferenceObjective.h
	AmplitudeObjective.h
    DynamicalSystemLimiter.h
    NeuronNetwork.h
    NetworkGAOptimizer.h
    DESTINATION ${PROJECT_NAME}/${PROJECT_VERSION}
)
