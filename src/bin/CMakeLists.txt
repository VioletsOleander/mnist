add_executable(mnist_show_data show_data.cpp)
target_link_libraries(mnist_show_data PRIVATE mnist)

add_executable(mnist_train train.cpp)
target_link_libraries(mnist_train PRIVATE mnist)

add_executable(mnist_test "test.cpp")
target_link_libraries(mnist_test PRIVATE mnist)
