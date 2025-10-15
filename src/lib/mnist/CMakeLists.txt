add_library(mnist STATIC
    data/dataset/dataset.cpp
    utils/utils.cpp
)

target_include_directories(mnist PUBLIC ${CMAKE_SOURCE_DIR}/src/lib ${OpenCV_INCLUDE_DIRS})
target_link_directories(mnist PUBLIC ${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib)

target_link_libraries(mnist PUBLIC ${TORCH_LIBRARIES} Arrow::arrow_shared Parquet::parquet_shared ${OpenCV_LIBS})

target_link_options(mnist PUBLIC -fuse-ld=lld)

