cmake_minimum_required(VERSION 3.8)
project(demo2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

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

install(TARGETS realsense_rgbd_viewer_cpp
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES "${REALSENSE_VENDOR_LIBRARY}"
  DESTINATION lib/${PROJECT_NAME}
  RENAME librealsense2.so.2.58
)

install(PROGRAMS
  src/demo.py
  src/test.py
  src/body_yolo.py
  src/arm_driver.py
  src/position_to_angle.py
  src/position_to_angle_agx.py
  src/ik_test_publisher.py
  src/demo2/arm_mediapipe3d.py
  src/demo2/arm_yolo26s_v2.py
  src/demo2/depth_arm_controller.py
  src/demo2/depth_arm_follower.py
  src/demo2/depth_pose_detector.py
  src/demo2/position_to_angle_v2.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

install(FILES
  src/arm_and_revo2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(FILES
  model/yolo26n-pose.pt
  model/yolo26s-pose.pt
  DESTINATION share/${PROJECT_NAME}/model
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  # Keep upstream librealsense headers out of project-owned style checks.
  set(AMENT_LINT_AUTO_FILE_EXCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
  )
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(
    pose_mapping test/test_pose_mapping.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    pose_retargeting_3d test/test_pose_retargeting_3d.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    depth_arm_geometry test/test_depth_arm_geometry.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    depth_pose_interface test/test_depth_pose_interface.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(
    person_camera_calibration test/test_person_camera_calibration.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
endif()

ament_package()
