cmake_minimum_required(VERSION 3.8)
project(demo2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS calib3d core dnn highgui imgproc video)

# Use the installed PyTorch runtime's LibTorch package for C++ inference.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path, end='')"
  RESULT_VARIABLE TORCH_PREFIX_RESULT
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  ERROR_QUIET
)
if(NOT TORCH_PREFIX_RESULT EQUAL 0 OR TORCH_CMAKE_PREFIX STREQUAL "")
  message(FATAL_ERROR
    "PyTorch/LibTorch was not found. Install the CPU torch runtime before "
    "building demo2."
  )
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
set(REALSENSE_VENDOR_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/librealsense2"
)
set(REALSENSE_VENDOR_LIBRARY
  "${REALSENSE_VENDOR_ROOT}/lib/librealsense2.so.2.58.2"
)
if(NOT EXISTS "${REALSENSE_VENDOR_LIBRARY}")
  message(FATAL_ERROR "Vendored librealsense2 library is missing")
endif()

add_library(realsense2_vendor SHARED IMPORTED GLOBAL)
set_target_properties(realsense2_vendor PROPERTIES
  IMPORTED_LOCATION "${REALSENSE_VENDOR_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${REALSENSE_VENDOR_ROOT}/include"
)

add_executable(realsense_rgbd_viewer_cpp src/realsense_rgbd_viewer.cpp)
target_compile_features(realsense_rgbd_viewer_cpp PRIVATE cxx_std_17)
ament_target_dependencies(realsense_rgbd_viewer_cpp rclcpp sensor_msgs)
target_link_libraries(realsense_rgbd_viewer_cpp
  realsense2_vendor
  ${OpenCV_LIBS}
)
set_target_properties(realsense_rgbd_viewer_cpp PROPERTIES
  BUILD_RPATH "${REALSENSE_VENDOR_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN"
)

add_executable(depth_pose_detector_cpp src/depth_pose_detector.cpp)
target_compile_features(depth_pose_detector_cpp PRIVATE cxx_std_17)
ament_target_dependencies(depth_pose_detector_cpp rclcpp sensor_msgs)
target_link_libraries(depth_pose_detector_cpp
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)
set_target_properties(depth_pose_detector_cpp PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

install(TARGETS realsense_rgbd_viewer_cpp depth_pose_detector_cpp
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES "${REALSENSE_VENDOR_LIBRARY}"
  DESTINATION lib/${PROJECT_NAME}
  RENAME librealsense2.so.2.58
)

install(PROGRAMS
  src/demo2/depth_arm_controller.py
  src/demo2/dual_joint_state_publisher.py
  src/demo2/pyagxarm_driver.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(FILES
  model/yolo26n-pose.torchscript
  DESTINATION share/${PROJECT_NAME}/model
)

if(BUILD_TESTING)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_uncrustify REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)
  file(GLOB DEMO2_PYTHON_LINT_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/launch/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/demo2/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/*.py"
  )
  ament_flake8(${DEMO2_PYTHON_LINT_FILES})
  ament_pep257(${DEMO2_PYTHON_LINT_FILES})
  ament_uncrustify(
    src/depth_pose_detector.cpp
    src/realsense_rgbd_viewer.cpp
  )
  ament_lint_cmake(CMakeLists.txt)
  ament_xmllint(package.xml)
  ament_add_pytest_test(
    arm_geometry test/test_arm_geometry.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    controller_interface test/test_controller_interface.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    nero_direction_ik test/test_nero_direction_ik.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    person_camera_calibration test/test_person_camera_calibration.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    tracking_stability test/test_tracking_stability.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    pyagxarm_driver test/test_pyagxarm_driver.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
endif()

ament_package()
