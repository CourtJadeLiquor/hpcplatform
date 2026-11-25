foreach(sample_source IN ITEMS
    00_hello_tensor
    01_memory_copy
    02_ops_pipeline)
  add_executable(${sample_source} ${sample_source}.cpp)
  target_link_libraries(${sample_source} PRIVATE hpcplatform)
endforeach()

if(hpcplatform_WITH_CUDA)
  add_executable(04_cuda_pipeline 04_cuda_pipeline.cpp)
  target_link_libraries(04_cuda_pipeline PRIVATE hpcplatform)
endif()

if(hpcplatform_WITH_OPENCV)
  add_executable(03_opencv_backend 03_opencv_backend.cpp)
  target_link_libraries(03_opencv_backend PRIVATE hpcplatform)
endif()

if(hpcplatform_WITH_YAML)
  add_executable(05_config_operator 05_config_operator.cpp)
  target_link_libraries(05_config_operator PRIVATE hpcplatform)
endif()

# PNG loading/saving for the cross-runtime image pipeline needs imgcodecs on top
# of the core/imgproc components the library already links.
if(hpcplatform_WITH_CUDA AND hpcplatform_WITH_OPENCV)
  find_package(OpenCV 4 REQUIRED COMPONENTS imgcodecs)
  add_executable(06_image_cuda_pipeline 06_image_cuda_pipeline.cpp)
  target_link_libraries(06_image_cuda_pipeline PRIVATE hpcplatform opencv_imgcodecs)
  target_compile_definitions(06_image_cuda_pipeline PRIVATE
    hpcplatform_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
