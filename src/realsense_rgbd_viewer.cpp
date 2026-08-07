#include <librealsense2/rs.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace
{

constexpr char kWindowName[] = "RealSense RGB | Depth | Fusion (C++)";

std::tuple<int, int, int> parse_profile(const std::string & value)
{
  std::istringstream stream(value);
  int width = 0;
  int height = 0;
  int fps = 0;
  char first_separator = 0;
  char second_separator = 0;
  if (!(stream >> width >> first_separator >> height >> second_separator >> fps) ||
    (first_separator != 'x' && first_separator != 'X') ||
    (second_separator != 'x' && second_separator != 'X') ||
    width <= 0 || height <= 0 || fps <= 0)
  {
    throw std::invalid_argument(
            "invalid stream profile '" + value + "'; expected WIDTHxHEIGHTxFPS");
  }
  return {width, height, fps};
}

std::string trim_slashes(std::string value)
{
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  return value;
}

std::string strip_quotes(std::string value)
{
  value.erase(
    std::remove_if(
      value.begin(), value.end(), [](char character) {
        return character == '\'' || character == '"';
      }),
    value.end());
  return value;
}

std::string topic_prefix(const std::string & camera_namespace, const std::string & camera_name)
{
  const auto clean_namespace = trim_slashes(camera_namespace);
  const auto clean_name = trim_slashes(camera_name);
  std::string result;
  if (!clean_namespace.empty()) {
    result += "/" + clean_namespace;
  }
  if (!clean_name.empty()) {
    result += "/" + clean_name;
  }
  return result.empty() ? "/camera/camera" : result;
}

}  // namespace

class RealsenseRgbdViewer : public rclcpp::Node
{
public:
  RealsenseRgbdViewer()
  : Node("realsense_rgbd_viewer_cpp"), running_(true)
  {
    camera_namespace_ = declare_parameter<std::string>("camera_namespace", "camera");
    camera_name_ = declare_parameter<std::string>("camera_name", "camera");
    serial_ = strip_quotes(declare_parameter<std::string>("serial_no", ""));
    color_profile_ = declare_parameter<std::string>("color_profile", "640x480x30");
    depth_profile_ = declare_parameter<std::string>("depth_profile", "640x480x30");
    initial_reset_pending_ = declare_parameter<bool>("initial_reset", false);
    spatial_filter_enabled_ = declare_parameter<bool>("spatial_filter_enabled", false);
    temporal_filter_enabled_ = declare_parameter<bool>("temporal_filter_enabled", true);
    show_gui_ = declare_parameter<bool>("show_gui", true);
    publish_composite_ = declare_parameter<bool>("publish_composite", true);
    min_depth_m_ = declare_parameter<double>("min_depth_m", 0.15);
    max_depth_m_ = declare_parameter<double>("max_depth_m", 5.0);
    fusion_alpha_ = declare_parameter<double>("fusion_alpha", 0.45);
    output_topic_ = declare_parameter<std::string>("output_topic", "/realsense/rgbd_view");
    if (max_depth_m_ <= min_depth_m_) {
      throw std::invalid_argument("max_depth_m must be greater than min_depth_m");
    }
    fusion_alpha_ = std::clamp(fusion_alpha_, 0.0, 1.0);
    std::tie(color_width_, color_height_, color_fps_) = parse_profile(color_profile_);
    std::tie(depth_width_, depth_height_, depth_fps_) = parse_profile(depth_profile_);

    const auto prefix = topic_prefix(camera_namespace_, camera_name_);
    frame_id_ = trim_slashes(camera_name_) + "_color_optical_frame";
    if (frame_id_ == "_color_optical_frame") {
      frame_id_ = "camera_color_optical_frame";
    }
    const auto qos = rclcpp::SensorDataQoS();
    color_publisher_ = create_publisher<sensor_msgs::msg::Image>(
      prefix + "/color/image_raw", qos);
    depth_publisher_ = create_publisher<sensor_msgs::msg::Image>(
      prefix + "/aligned_depth_to_color/image_raw", qos);
    info_publisher_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      prefix + "/color/camera_info", qos);
    composite_publisher_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, qos);

    RCLCPP_INFO(
      get_logger(),
      "C++ RealSense RGB-D node configured: %dx%d@%d under %s",
      color_width_, color_height_, color_fps_, prefix.c_str());
    worker_ = std::thread(&RealsenseRgbdViewer::capture_loop, this);
  }

  ~RealsenseRgbdViewer() override
  {
    running_.store(false);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  bool matching_device_present()
  {
    rs2::context context;
    const auto devices = context.query_devices();
    if (devices.size() == 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "No RealSense detected; waiting for a USB 3 connection");
      return false;
    }
    if (serial_.empty()) {
      return true;
    }
    for (auto && device : devices) {
      if (device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) &&
        serial_ == device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))
      {
        return true;
      }
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "Requested RealSense serial '%s' is not connected", serial_.c_str());
    return false;
  }

  void reset_selected_device()
  {
    rs2::context context;
    for (auto && device : context.query_devices()) {
      if (serial_.empty() ||
        (device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) &&
        serial_ == device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)))
      {
        device.hardware_reset();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return;
      }
    }
  }

  sensor_msgs::msg::CameraInfo camera_info_from(const rs2_intrinsics & intrinsics) const
  {
    sensor_msgs::msg::CameraInfo message;
    message.width = static_cast<uint32_t>(intrinsics.width);
    message.height = static_cast<uint32_t>(intrinsics.height);
    message.distortion_model = "plumb_bob";
    message.d.assign(intrinsics.coeffs, intrinsics.coeffs + 5);
    message.k = {
      intrinsics.fx, 0.0, intrinsics.ppx,
      0.0, intrinsics.fy, intrinsics.ppy,
      0.0, 0.0, 1.0};
    message.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    message.p = {
      intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
      0.0, intrinsics.fy, intrinsics.ppy, 0.0,
      0.0, 0.0, 1.0, 0.0};
    return message;
  }

  sensor_msgs::msg::Image image_message(
    const cv::Mat & image, const std::string & encoding, const rclcpp::Time & stamp) const
  {
    sensor_msgs::msg::Image message;
    message.header.stamp = stamp;
    message.header.frame_id = frame_id_;
    message.height = static_cast<uint32_t>(image.rows);
    message.width = static_cast<uint32_t>(image.cols);
    message.encoding = encoding;
    message.is_bigendian = false;
    message.step = static_cast<uint32_t>(image.step);
    const auto bytes = static_cast<std::size_t>(image.rows) * image.step;
    message.data.resize(bytes);
    std::memcpy(message.data.data(), image.data, bytes);
    return message;
  }

  cv::Mat compose_view(const cv::Mat & color, const cv::Mat & raw_depth, float depth_scale) const
  {
    cv::Mat normalized(raw_depth.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat valid(raw_depth.size(), CV_8UC1, cv::Scalar(0));
    const auto denominator = max_depth_m_ - min_depth_m_;
    for (int row = 0; row < raw_depth.rows; ++row) {
      const auto * source = raw_depth.ptr<uint16_t>(row);
      auto * target = normalized.ptr<uint8_t>(row);
      auto * mask = valid.ptr<uint8_t>(row);
      for (int column = 0; column < raw_depth.cols; ++column) {
        const double meters = static_cast<double>(source[column]) * depth_scale;
        if (source[column] != 0 && meters >= min_depth_m_ && meters <= max_depth_m_) {
          target[column] = static_cast<uint8_t>(std::lround(
              (max_depth_m_ - meters) / denominator * 255.0));
          mask[column] = 255;
        }
      }
    }

    cv::Mat depth_color;
    cv::applyColorMap(normalized, depth_color, cv::COLORMAP_TURBO);
    cv::Mat invalid;
    cv::bitwise_not(valid, invalid);
    depth_color.setTo(cv::Scalar(0, 0, 0), invalid);

    cv::Mat fusion;
    cv::addWeighted(color, 1.0 - fusion_alpha_, depth_color, fusion_alpha_, 0.0, fusion);
    color.copyTo(fusion, invalid);

    std::vector<cv::Mat> panels{color.clone(), depth_color, fusion};
    const std::vector<std::string> labels{"RGB", "ALIGNED DEPTH", "RGB + DEPTH"};
    for (std::size_t index = 0; index < panels.size(); ++index) {
      cv::rectangle(panels[index], cv::Point(0, 0), cv::Point(220, 36), cv::Scalar(0, 0, 0), -1);
      cv::putText(
        panels[index], labels[index], cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    std::ostringstream legend;
    legend.precision(2);
    legend << std::fixed << min_depth_m_ << "m near  <->  " << max_depth_m_ << "m far";
    cv::putText(
      panels[1], legend.str(), cv::Point(10, color.rows - 12), cv::FONT_HERSHEY_SIMPLEX,
      0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    const cv::Mat separator(color.rows, 4, CV_8UC3, cv::Scalar(230, 230, 230));
    cv::Mat combined;
    cv::hconcat(
      std::vector<cv::Mat>{panels[0], separator, panels[1], separator, panels[2]}, combined);
    return combined;
  }

  void publish_frames(
    const cv::Mat & color, const cv::Mat & depth, const cv::Mat & composite,
    sensor_msgs::msg::CameraInfo camera_info)
  {
    const auto stamp = now();
    camera_info.header.stamp = stamp;
    camera_info.header.frame_id = frame_id_;
    info_publisher_->publish(camera_info);
    depth_publisher_->publish(image_message(depth, "16UC1", stamp));
    // Color is the detector trigger, so publish it after its matching depth.
    color_publisher_->publish(image_message(color, "bgr8", stamp));
    if (publish_composite_) {
      composite_publisher_->publish(image_message(composite, "bgr8", stamp));
    }
  }

  void run_camera_session()
  {
    rs2::pipeline pipeline;
    rs2::config config;
    if (!serial_.empty()) {
      config.enable_device(serial_);
    }
    config.enable_stream(
      RS2_STREAM_COLOR, color_width_, color_height_, RS2_FORMAT_BGR8, color_fps_);
    config.enable_stream(
      RS2_STREAM_DEPTH, depth_width_, depth_height_, RS2_FORMAT_Z16, depth_fps_);
    const auto profile = pipeline.start(config);
    const auto device = profile.get_device();
    const auto active_serial =
      device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) ?
      std::string(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) : "unknown";
    const auto physical_port =
      device.supports(RS2_CAMERA_INFO_PHYSICAL_PORT) ?
      std::string(device.get_info(RS2_CAMERA_INFO_PHYSICAL_PORT)) : "unknown";
    const auto usb_type =
      device.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ?
      std::string(device.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)) :
      "unknown";
    const auto depth_scale =
      device.first<rs2::depth_sensor>().get_depth_scale();
    const auto video_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    const auto camera_info = camera_info_from(video_profile.get_intrinsics());
    rs2::align align_to_color(RS2_STREAM_COLOR);
    rs2::spatial_filter spatial_filter;
    rs2::temporal_filter temporal_filter;
    bool window_created = false;
    std::size_t frame_count = 0;
    auto rate_start = std::chrono::steady_clock::now();
    RCLCPP_INFO(
      get_logger(),
      "Local USB RealSense connected: serial=%s usb=%s port=%s; "
      "C++ aligned RGB-D is running",
      active_serial.c_str(), usb_type.c_str(), physical_port.c_str());

    try {
      while (running_.load() && rclcpp::ok()) {
        const auto frames = pipeline.wait_for_frames(1000);
        const auto aligned = align_to_color.process(frames);
        const auto color_frame = aligned.get_color_frame();
        rs2::depth_frame depth_frame = aligned.get_depth_frame();
        if (!color_frame || !depth_frame) {
          continue;
        }
        if (spatial_filter_enabled_) {
          depth_frame = spatial_filter.process(depth_frame).as<rs2::depth_frame>();
        }
        if (temporal_filter_enabled_) {
          depth_frame = temporal_filter.process(depth_frame).as<rs2::depth_frame>();
        }
        cv::Mat color(
          color_frame.get_height(), color_frame.get_width(), CV_8UC3,
          const_cast<void *>(color_frame.get_data()), color_frame.get_stride_in_bytes());
        cv::Mat depth(
          depth_frame.get_height(), depth_frame.get_width(), CV_16UC1,
          const_cast<void *>(depth_frame.get_data()), depth_frame.get_stride_in_bytes());
        cv::Mat composite;
        if (show_gui_ || publish_composite_) {
          composite = compose_view(color, depth, depth_scale);
        }
        publish_frames(color, depth, composite, camera_info);
        if (show_gui_) {
          if (!window_created) {
            cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
            cv::resizeWindow(kWindowName, 1500, 500);
            window_created = true;
          }
          cv::imshow(kWindowName, composite);
          cv::waitKey(1);
        }

        ++frame_count;
        const auto now_time = std::chrono::steady_clock::now();
        const auto seconds = std::chrono::duration<double>(now_time - rate_start).count();
        if (seconds >= 3.0) {
          RCLCPP_INFO(get_logger(), "C++ RGB-D processing rate: %.1f FPS", frame_count / seconds);
          frame_count = 0;
          rate_start = now_time;
        }
      }
    } catch (...) {
      pipeline.stop();
      if (window_created) {
        cv::destroyWindow(kWindowName);
      }
      throw;
    }
    pipeline.stop();
    if (window_created) {
      cv::destroyWindow(kWindowName);
    }
  }

  void capture_loop()
  {
    while (running_.load() && rclcpp::ok()) {
      try {
        if (!matching_device_present()) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
        if (initial_reset_pending_) {
          reset_selected_device();
          initial_reset_pending_ = false;
          continue;
        }
        run_camera_session();
      } catch (const rs2::error & error) {
        if (running_.load() && rclcpp::ok()) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 3000, "RealSense C++ error: %s", error.what());
        }
      } catch (const std::exception & error) {
        if (running_.load() && rclcpp::ok()) {
          RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 3000, "C++ RGB-D error: %s", error.what());
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  std::atomic<bool> running_;
  std::thread worker_;
  std::string camera_namespace_;
  std::string camera_name_;
  std::string serial_;
  std::string color_profile_;
  std::string depth_profile_;
  std::string output_topic_;
  std::string frame_id_;
  bool initial_reset_pending_;
  bool spatial_filter_enabled_;
  bool temporal_filter_enabled_;
  bool show_gui_;
  bool publish_composite_;
  double min_depth_m_;
  double max_depth_m_;
  double fusion_alpha_;
  int color_width_;
  int color_height_;
  int color_fps_;
  int depth_width_;
  int depth_height_;
  int depth_fps_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr composite_publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RealsenseRgbdViewer>());
  rclcpp::shutdown();
  return 0;
}
