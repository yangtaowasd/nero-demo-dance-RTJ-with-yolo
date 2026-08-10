#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/script.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/channel_float32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

constexpr std::size_t kLandmarkCount = 8;
constexpr std::array<int, kLandmarkCount> kYoloLandmarkIds{{5, 7, 9, 11, 6, 8, 10, 12}};
constexpr std::array<const char *, kLandmarkCount> kLabels{{
  "LS", "LE", "LW", "LH", "RS", "RE", "RW", "RH"}};
constexpr std::array<std::pair<int, int>, 8> kConnections{{
  {0, 1}, {1, 2}, {4, 5}, {5, 6}, {0, 4}, {0, 3}, {4, 7}, {3, 7}}};
constexpr std::array<int, 6> kLeftRequired{{0, 3, 4, 7, 1, 2}};
constexpr std::array<int, 6> kRightRequired{{0, 3, 4, 7, 5, 6}};

double stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

float median(std::vector<float> values)
{
  if (values.empty()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  const float upper = *middle;
  if (values.size() % 2 != 0) {
    return upper;
  }
  const auto lower = std::max_element(values.begin(), middle);
  return 0.5F * (upper + *lower);
}

float quantile_quarter(std::vector<float> values)
{
  if (values.empty()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const std::size_t index = (values.size() - 1) / 4;
  const auto position = values.begin() + static_cast<std::ptrdiff_t>(index);
  std::nth_element(values.begin(), position, values.end());
  return *position;
}

float box_iou(const cv::Rect2f & first, const cv::Rect2f & second)
{
  const cv::Rect2f intersection = first & second;
  const float intersection_area = std::max(intersection.area(), 0.0F);
  const float union_area = first.area() + second.area() - intersection_area;
  return union_area > 1e-9F ? intersection_area / union_area : 0.0F;
}

float normalized_center_distance(const cv::Rect2f & first, const cv::Rect2f & second)
{
  const cv::Point2f delta =
    (first.tl() + first.br()) * 0.5F - (second.tl() + second.br()) * 0.5F;
  const float first_diagonal = std::hypot(first.width, first.height);
  const float second_diagonal = std::hypot(second.width, second.height);
  return cv::norm(delta) / std::max({first_diagonal, second_diagonal, 1.0F});
}

sensor_msgs::msg::Image image_message(
  const std_msgs::msg::Header & header, const cv::Mat & image)
{
  sensor_msgs::msg::Image message;
  message.header = header;
  message.height = static_cast<uint32_t>(image.rows);
  message.width = static_cast<uint32_t>(image.cols);
  message.encoding = "bgr8";
  message.is_bigendian = false;
  message.step = static_cast<uint32_t>(image.cols * 3);
  const cv::Mat contiguous = image.isContinuous() ? image : image.clone();
  const std::size_t bytes = static_cast<std::size_t>(message.step) * message.height;
  message.data.resize(bytes);
  std::memcpy(message.data.data(), contiguous.data, bytes);
  return message;
}

}  // namespace

class DepthPoseDetectorCpp : public rclcpp::Node
{
public:
  DepthPoseDetectorCpp()
  : Node("depth_pose_detector_cpp"), running_(true)
  {
    color_topic_ = declare_parameter<std::string>(
      "color_topic", "/camera/camera/color/image_raw");
    depth_topic_ = declare_parameter<std::string>(
      "aligned_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    camera_info_topic_ = declare_parameter<std::string>(
      "camera_info_topic", "/camera/camera/color/camera_info");
    landmarks_topic_ = declare_parameter<std::string>(
      "landmarks_topic", "/realsense/landmarks_3d");
    debug_topic_ = declare_parameter<std::string>(
      "debug_image_topic", "/realsense/arm_pose_debug");
    model_path_ = declare_parameter<std::string>("model_path", "");
    inference_size_ = declare_parameter<int>("inference_size", 384);
    torch_threads_ = declare_parameter<int>("torch_threads", 1);
    person_confidence_ = declare_parameter<double>("person_confidence", 0.30);
    min_landmark_confidence_ = declare_parameter<double>("min_landmark_confidence", 0.35);
    target_lock_enabled_ = declare_parameter<bool>("target_lock_enabled", true);
    target_lock_min_iou_ = declare_parameter<double>("target_lock_min_iou", 0.05);
    target_lock_max_distance_ = declare_parameter<double>(
      "target_lock_max_center_distance_ratio", 1.25);
    target_lock_max_missed_ = declare_parameter<int>("target_lock_max_missed_frames", 8);
    keypoint_alpha_ = declare_parameter<double>("keypoint_smoothing_alpha", 0.55);
    depth_scale_ = declare_parameter<double>("depth_uint16_scale", 0.001);
    depth_radius_ = declare_parameter<int>("depth_window_radius", 4);
    min_valid_depth_pixels_ = declare_parameter<int>("min_valid_depth_pixels", 4);
    depth_cluster_tolerance_ = declare_parameter<double>("depth_cluster_tolerance_m", 0.08);
    min_depth_ = declare_parameter<double>("min_depth_m", 0.15);
    max_depth_ = declare_parameter<double>("max_depth_m", 8.0);
    sync_tolerance_ = declare_parameter<double>("sync_tolerance_sec", 0.02);
    sync_wait_ = declare_parameter<double>("sync_wait_sec", 0.02);
    publish_debug_ = declare_parameter<bool>("publish_debug_image", true);
    show_gui_ = declare_parameter<bool>("show_gui", true);

    if (model_path_.empty()) {
      throw std::invalid_argument("model_path must point to a TorchScript YOLO pose model");
    }
    if (inference_size_ < 128 || inference_size_ % 32 != 0) {
      throw std::invalid_argument("inference_size must be >=128 and divisible by 32");
    }
    if (min_depth_ < 0.0 || max_depth_ <= min_depth_) {
      throw std::invalid_argument("depth range must satisfy 0 <= min_depth_m < max_depth_m");
    }
    if (sync_tolerance_ < 0.0 || sync_wait_ < 0.0) {
      throw std::invalid_argument("RGB-D synchronization limits must be non-negative");
    }
    previous_depths_.fill(std::numeric_limits<float>::quiet_NaN());
    keypoint_alpha_ = std::clamp(keypoint_alpha_, 0.0, 1.0);
    at::set_num_threads(std::max(torch_threads_, 1));
    at::set_num_interop_threads(1);
    module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path_));
    module_->eval();
    warm_up();

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(2);
    landmarks_publisher_ = create_publisher<sensor_msgs::msg::PointCloud>(
      landmarks_topic_, sensor_qos);
    debug_publisher_ = create_publisher<sensor_msgs::msg::Image>(debug_topic_, sensor_qos);
    color_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      color_topic_, sensor_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        {
          std::lock_guard<std::mutex> lock(data_mutex_);
          latest_color_ = std::move(message);
          ++color_sequence_;
        }
        data_ready_.notify_one();
      });
    depth_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_, sensor_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        {
          std::lock_guard<std::mutex> lock(data_mutex_);
          depth_frames_.push_back(std::move(message));
          while (depth_frames_.size() > 8) {
            depth_frames_.pop_front();
          }
        }
        data_ready_.notify_one();
      });
    info_subscription_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, sensor_qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        {
          std::lock_guard<std::mutex> lock(data_mutex_);
          camera_info_ = std::move(message);
        }
        data_ready_.notify_one();
      });

    worker_ = std::thread(&DepthPoseDetectorCpp::processing_loop, this);
    RCLCPP_INFO(
      get_logger(),
      "C++ RGB-D YOLO ready: model=%s input=%dx%d torch_threads=%d",
      model_path_.c_str(), inference_size_, inference_size_, std::max(torch_threads_, 1));
  }

  ~DepthPoseDetectorCpp() override
  {
    running_.store(false);
    data_ready_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (show_gui_) {
      cv::destroyAllWindows();
    }
  }

private:
  struct FrameBundle
  {
    std::size_t sequence;
    sensor_msgs::msg::Image::ConstSharedPtr color;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    double sync_delta;
  };

  struct Detection
  {
    cv::Rect2f box;
    float confidence;
    float quality;
    std::array<cv::Point2f, kLandmarkCount> pixels;
    std::array<float, kLandmarkCount> landmark_confidence;
  };

  void warm_up()
  {
    c10::InferenceMode guard;
    auto input = torch::zeros(
      {1, 3, inference_size_, inference_size_}, torch::TensorOptions().dtype(torch::kFloat32));
    (void)module_->forward({input});
  }

  std::optional<FrameBundle> take_frame()
  {
    std::unique_lock<std::mutex> lock(data_mutex_);
    data_ready_.wait_for(
      lock, std::chrono::milliseconds(30), [this]() {
        return !running_.load() ||
        (latest_color_ && color_sequence_ != processed_color_sequence_);
      });
    if (!running_.load() || !latest_color_ || color_sequence_ == processed_color_sequence_) {
      return std::nullopt;
    }

    auto waited_sequence = color_sequence_;
    auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(sync_wait_));
    while (running_.load()) {
      if (color_sequence_ != waited_sequence) {
        waited_sequence = color_sequence_;
        deadline = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(sync_wait_));
      }
      if (latest_color_ && camera_info_ && !depth_frames_.empty()) {
        const double color_stamp = stamp_seconds(latest_color_->header.stamp);
        const auto closest = std::min_element(
          depth_frames_.begin(), depth_frames_.end(),
          [color_stamp](const auto & first, const auto & second) {
            return std::abs(stamp_seconds(first->header.stamp) - color_stamp) <
            std::abs(stamp_seconds(second->header.stamp) - color_stamp);
          });
        const double delta =
          std::abs(stamp_seconds((*closest)->header.stamp) - color_stamp);
        if (delta <= sync_tolerance_ || std::chrono::steady_clock::now() >= deadline) {
          FrameBundle bundle{
            color_sequence_, latest_color_, *closest, camera_info_, delta};
          processed_color_sequence_ = color_sequence_;
          return bundle;
        }
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        processed_color_sequence_ = color_sequence_;
        return std::nullopt;
      }
      data_ready_.wait_until(lock, deadline);
    }
    return std::nullopt;
  }

  static cv::Mat color_mat(const sensor_msgs::msg::Image & message)
  {
    const std::string & encoding = message.encoding;
    int channels = 0;
    if (encoding == "bgr8" || encoding == "rgb8") {
      channels = 3;
    } else if (encoding == "bgra8" || encoding == "rgba8") {
      channels = 4;
    } else {
      throw std::invalid_argument("unsupported color encoding: " + encoding);
    }
    const std::size_t required_step = static_cast<std::size_t>(message.width) * channels;
    if (message.step < required_step ||
      message.data.size() < static_cast<std::size_t>(message.step) * message.height)
    {
      throw std::invalid_argument("truncated color image");
    }
    cv::Mat wrapped(
      static_cast<int>(message.height), static_cast<int>(message.width),
      channels == 3 ? CV_8UC3 : CV_8UC4,
      const_cast<unsigned char *>(message.data.data()), message.step);
    cv::Mat bgr;
    if (encoding == "bgr8") {
      bgr = wrapped.clone();
    } else if (encoding == "rgb8") {
      cv::cvtColor(wrapped, bgr, cv::COLOR_RGB2BGR);
    } else if (encoding == "bgra8") {
      cv::cvtColor(wrapped, bgr, cv::COLOR_BGRA2BGR);
    } else {
      cv::cvtColor(wrapped, bgr, cv::COLOR_RGBA2BGR);
    }
    return bgr;
  }

  std::vector<Detection> infer(const cv::Mat & frame)
  {
    const double scale = std::min(
      static_cast<double>(inference_size_) / frame.cols,
      static_cast<double>(inference_size_) / frame.rows);
    const int resized_width = static_cast<int>(std::round(frame.cols * scale));
    const int resized_height = static_cast<int>(std::round(frame.rows * scale));
    const int pad_x = (inference_size_ - resized_width) / 2;
    const int pad_y = (inference_size_ - resized_height) / 2;
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat letterboxed(
      inference_size_, inference_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    cv::Mat blob = cv::dnn::blobFromImage(
      letterboxed, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

    c10::InferenceMode guard;
    auto input = torch::from_blob(
      blob.ptr<float>(), {1, 3, inference_size_, inference_size_},
      torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor output = module_->forward({input}).toTensor().to(torch::kCPU).contiguous();
    if (output.dim() != 3 || output.size(0) != 1 || output.size(2) < 57) {
      throw std::runtime_error("unexpected TorchScript output shape; expected 1xNx57");
    }
    const auto values = output.accessor<float, 3>();
    std::vector<Detection> detections;
    for (int64_t row = 0; row < output.size(1); ++row) {
      const float confidence = values[0][row][4];
      if (!std::isfinite(confidence) || confidence < person_confidence_) {
        continue;
      }
      const float x1 = static_cast<float>((values[0][row][0] - pad_x) / scale);
      const float y1 = static_cast<float>((values[0][row][1] - pad_y) / scale);
      const float x2 = static_cast<float>((values[0][row][2] - pad_x) / scale);
      const float y2 = static_cast<float>((values[0][row][3] - pad_y) / scale);
      Detection detection;
      detection.box = cv::Rect2f(x1, y1, std::max(x2 - x1, 1.0F), std::max(y2 - y1, 1.0F));
      detection.confidence = confidence;
      detection.quality = 0.0F;
      bool finite = true;
      for (std::size_t index = 0; index < kLandmarkCount; ++index) {
        const int base = 6 + 3 * kYoloLandmarkIds[index];
        const float x = static_cast<float>((values[0][row][base] - pad_x) / scale);
        const float y = static_cast<float>((values[0][row][base + 1] - pad_y) / scale);
        const float point_confidence = values[0][row][base + 2];
        detection.pixels[index] = cv::Point2f(x, y);
        detection.landmark_confidence[index] = point_confidence;
        detection.quality += point_confidence;
        finite = finite && std::isfinite(x) && std::isfinite(y) &&
          std::isfinite(point_confidence);
      }
      detection.quality /= static_cast<float>(kLandmarkCount);
      if (finite) {
        detections.push_back(detection);
      }
    }
    return detections;
  }

  std::optional<Detection> select_target(std::vector<Detection> detections)
  {
    if (detections.empty()) {
      if (tracked_box_) {
        ++missed_frames_;
        tracking_state_ = "coasting";
        if (missed_frames_ > target_lock_max_missed_) {
          tracked_box_.reset();
          previous_pixels_.reset();
          previous_depths_.fill(std::numeric_limits<float>::quiet_NaN());
          missed_frames_ = 0;
          tracking_state_ = "lost";
        }
      }
      return std::nullopt;
    }
    auto strongest = [&detections]() {
        return static_cast<std::size_t>(std::distance(
                 detections.begin(), std::max_element(
                   detections.begin(), detections.end(),
                   [](const auto & first, const auto & second) {
                     return first.quality < second.quality;
                   })));
      };
    std::size_t selected = strongest();
    bool continued = false;
    if (target_lock_enabled_ && tracked_box_) {
      float best_score = -std::numeric_limits<float>::infinity();
      bool matched = false;
      for (std::size_t index = 0; index < detections.size(); ++index) {
        const float overlap = box_iou(*tracked_box_, detections[index].box);
        const float distance = normalized_center_distance(*tracked_box_, detections[index].box);
        if (overlap < target_lock_min_iou_ && distance > target_lock_max_distance_) {
          continue;
        }
        const float score =
          0.60F * overlap + 0.25F * std::exp(-distance) + 0.15F * detections[index].quality;
        if (score > best_score) {
          best_score = score;
          selected = index;
          matched = true;
        }
      }
      if (!matched) {
        ++missed_frames_;
        tracking_state_ = "coasting";
        if (missed_frames_ <= target_lock_max_missed_) {
          return std::nullopt;
        }
        tracked_box_.reset();
        previous_pixels_.reset();
        previous_depths_.fill(std::numeric_limits<float>::quiet_NaN());
      } else {
        continued = true;
      }
    }
    Detection result = detections[selected];
    if (!tracked_box_) {
      previous_depths_.fill(std::numeric_limits<float>::quiet_NaN());
    }
    if (!target_lock_enabled_ || !tracked_box_) {
      tracked_box_ = result.box;
      missed_frames_ = 0;
      tracking_state_ = "acquired";
    } else {
      const float alpha = 0.65F;
      tracked_box_->x = alpha * result.box.x + (1.0F - alpha) * tracked_box_->x;
      tracked_box_->y = alpha * result.box.y + (1.0F - alpha) * tracked_box_->y;
      tracked_box_->width = alpha * result.box.width + (1.0F - alpha) * tracked_box_->width;
      tracked_box_->height = alpha * result.box.height + (1.0F - alpha) * tracked_box_->height;
      missed_frames_ = 0;
      tracking_state_ = continued ? "locked" : "acquired";
    }
    if (continued && previous_pixels_) {
      for (std::size_t index = 0; index < kLandmarkCount; ++index) {
        result.pixels[index] =
          static_cast<float>(keypoint_alpha_) * result.pixels[index] +
          static_cast<float>(1.0 - keypoint_alpha_) * (*previous_pixels_)[index];
      }
    }
    previous_pixels_ = result.pixels;
    return result;
  }

  float depth_at(
    const sensor_msgs::msg::Image & depth, const cv::Point2f & pixel,
    float previous_depth) const
  {
    const int center_x = static_cast<int>(std::lround(pixel.x));
    const int center_y = static_cast<int>(std::lround(pixel.y));
    if (center_x < 0 || center_x >= static_cast<int>(depth.width) ||
      center_y < 0 || center_y >= static_cast<int>(depth.height))
    {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const bool uint16 = depth.encoding == "16UC1" || depth.encoding == "mono16";
    const bool float32 = depth.encoding == "32FC1";
    if (!uint16 && !float32) {
      throw std::invalid_argument("unsupported depth encoding: " + depth.encoding);
    }
    const std::size_t item_size = uint16 ? sizeof(uint16_t) : sizeof(float);
    if (depth.step < depth.width * item_size ||
      depth.data.size() < static_cast<std::size_t>(depth.step) * depth.height)
    {
      throw std::invalid_argument("truncated depth image");
    }
    const auto read = [&](int x, int y) {
        const unsigned char * address = depth.data.data() +
          static_cast<std::size_t>(y) * depth.step + static_cast<std::size_t>(x) * item_size;
        if (uint16) {
          uint16_t raw = 0;
          std::memcpy(&raw, address, sizeof(raw));
          if (depth.is_bigendian) {
            raw = static_cast<uint16_t>((raw >> 8U) | (raw << 8U));
          }
          return static_cast<float>(raw * depth_scale_);
        }
        float raw = 0.0F;
        std::memcpy(&raw, address, sizeof(raw));
        return raw;
      };
    std::vector<float> valid;
    for (int y = std::max(center_y - depth_radius_, 0);
      y <= std::min(center_y + depth_radius_, static_cast<int>(depth.height) - 1); ++y)
    {
      for (int x = std::max(center_x - depth_radius_, 0);
        x <= std::min(center_x + depth_radius_, static_cast<int>(depth.width) - 1); ++x)
      {
        const float value = read(x, y);
        if (std::isfinite(value) && value >= min_depth_ && value <= max_depth_) {
          valid.push_back(value);
        }
      }
    }
    if (valid.size() < static_cast<std::size_t>(std::max(min_valid_depth_pixels_, 1))) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const auto cluster_around = [&valid, this](float anchor) {
        const float tolerance = std::max(
          static_cast<float>(depth_cluster_tolerance_), 0.015F * anchor);
        std::vector<float> cluster;
        cluster.reserve(valid.size());
        for (const float value : valid) {
          if (std::abs(value - anchor) <= tolerance) {
            cluster.push_back(value);
          }
        }
        return cluster;
      };
    std::vector<float> cluster;
    if (std::isfinite(previous_depth) &&
      previous_depth >= min_depth_ && previous_depth <= max_depth_)
    {
      cluster = cluster_around(previous_depth);
    }
    if (cluster.size() < static_cast<std::size_t>(std::max(min_valid_depth_pixels_, 1))) {
      cluster = cluster_around(quantile_quarter(valid));
    }
    if (cluster.size() < static_cast<std::size_t>(std::max(min_valid_depth_pixels_, 1))) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return median(std::move(cluster));
  }

  std::array<cv::Point3f, kLandmarkCount> reconstruct(
    const Detection & detection, const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & info)
  {
    if (depth.width != info.width || depth.height != info.height ||
      info.k[0] <= 0.0 || info.k[4] <= 0.0)
    {
      throw std::invalid_argument("depth dimensions/intrinsics do not match color");
    }
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
      info.k[0], 0.0, info.k[2], 0.0, info.k[4], info.k[5], 0.0, 0.0, 1.0);
    cv::Mat distortion(info.d, true);
    std::array<cv::Point3f, kLandmarkCount> points;
    for (std::size_t index = 0; index < kLandmarkCount; ++index) {
      if (detection.landmark_confidence[index] < min_landmark_confidence_) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        points[index] = cv::Point3f(nan, nan, nan);
        continue;
      }
      const float z = depth_at(depth, detection.pixels[index], previous_depths_[index]);
      if (!std::isfinite(z)) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        points[index] = cv::Point3f(nan, nan, nan);
        continue;
      }
      previous_depths_[index] = z;
      cv::Point2f normalized;
      const bool distorted = !info.d.empty() && std::any_of(
        info.d.begin(), info.d.end(), [](double value) {return std::abs(value) > 1e-12;});
      if (distorted) {
        std::vector<cv::Point2f> source{detection.pixels[index]};
        std::vector<cv::Point2f> result;
        cv::undistortPoints(source, result, camera_matrix, distortion);
        normalized = result.front();
      } else {
        normalized.x = static_cast<float>((detection.pixels[index].x - info.k[2]) / info.k[0]);
        normalized.y = static_cast<float>((detection.pixels[index].y - info.k[5]) / info.k[4]);
      }
      points[index] = cv::Point3f(normalized.x * z, normalized.y * z, z);
    }
    return points;
  }

  template<std::size_t Size>
  bool side_ready(
    const std::array<cv::Point3f, kLandmarkCount> & points,
    const std::array<float, kLandmarkCount> & confidence,
    const std::array<int, Size> & required) const
  {
    return std::all_of(
      required.begin(), required.end(), [&](int index) {
        const auto & point = points[static_cast<std::size_t>(index)];
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
        std::isfinite(confidence[static_cast<std::size_t>(index)]) &&
        confidence[static_cast<std::size_t>(index)] >= min_landmark_confidence_;
      });
  }

  template<std::size_t Size>
  std::string missing_required(
    const std::array<cv::Point3f, kLandmarkCount> & points,
    const std::array<float, kLandmarkCount> & confidence,
    const std::array<int, Size> & required) const
  {
    std::ostringstream result;
    bool first = true;
    for (const int raw_index : required) {
      const auto index = static_cast<std::size_t>(raw_index);
      const auto & point = points[index];
      std::string reason;
      if (!std::isfinite(confidence[index]) ||
        confidence[index] < min_landmark_confidence_)
      {
        std::ostringstream detail;
        detail.precision(2);
        detail << kLabels[index] << "(conf=" << std::fixed << confidence[index] << ")";
        reason = detail.str();
      } else if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z))
      {
        reason = std::string(kLabels[index]) + "(depth)";
      }
      if (!reason.empty()) {
        if (!first) {
          result << ",";
        }
        result << reason;
        first = false;
      }
    }
    return result.str();
  }

  sensor_msgs::msg::PointCloud point_cloud(
    const std_msgs::msg::Header & header,
    const std::optional<std::array<cv::Point3f, kLandmarkCount>> & points,
    const std::optional<std::array<float, kLandmarkCount>> & confidence) const
  {
    sensor_msgs::msg::PointCloud message;
    message.header = header;
    if (!points || !confidence) {
      return message;
    }
    message.points.resize(kLandmarkCount);
    for (std::size_t index = 0; index < kLandmarkCount; ++index) {
      message.points[index].x = (*points)[index].x;
      message.points[index].y = (*points)[index].y;
      message.points[index].z = (*points)[index].z;
    }
    sensor_msgs::msg::ChannelFloat32 confidence_channel;
    confidence_channel.name = "confidence";
    sensor_msgs::msg::ChannelFloat32 id_channel;
    id_channel.name = "landmark_id";
    sensor_msgs::msg::ChannelFloat32 valid_channel;
    valid_channel.name = "depth_valid";
    for (std::size_t index = 0; index < kLandmarkCount; ++index) {
      confidence_channel.values.push_back((*confidence)[index]);
      id_channel.values.push_back(static_cast<float>(index));
      const auto & point = (*points)[index];
      valid_channel.values.push_back(
        std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) ? 1.0F : 0.0F);
    }
    message.channels = {confidence_channel, id_channel, valid_channel};
    return message;
  }

  void annotate(
    cv::Mat & frame, const Detection & detection,
    const std::array<cv::Point3f, kLandmarkCount> & points,
    const std::string & status, bool valid) const
  {
    if (tracked_box_) {
      cv::rectangle(frame, *tracked_box_, cv::Scalar(255, 170, 40), 2);
      std::ostringstream label;
      label.precision(2);
      label << std::fixed << "YOLO person " << detection.confidence << " " << tracking_state_;
      cv::putText(
        frame, label.str(), cv::Point(
          static_cast<int>(tracked_box_->x), std::max(static_cast<int>(tracked_box_->y) - 7, 18)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 170, 40), 2, cv::LINE_AA);
    }
    for (const auto & connection : kConnections) {
      if (detection.landmark_confidence[connection.first] >= min_landmark_confidence_ &&
        detection.landmark_confidence[connection.second] >= min_landmark_confidence_)
      {
        cv::line(
          frame, detection.pixels[connection.first], detection.pixels[connection.second],
          cv::Scalar(70, 230, 100), 2, cv::LINE_AA);
      }
    }
    for (std::size_t index = 0; index < kLandmarkCount; ++index) {
      const bool confident =
        detection.landmark_confidence[index] >= min_landmark_confidence_;
      const bool depth_valid = std::isfinite(points[index].x) &&
        std::isfinite(points[index].y) && std::isfinite(points[index].z);
      const bool accepted = confident && depth_valid;
      const cv::Scalar color = accepted ? cv::Scalar(70, 230, 100) :
        (confident ? cv::Scalar(0, 80, 255) : cv::Scalar(0, 200, 255));
      cv::circle(frame, detection.pixels[index], 5, color, -1);
      std::ostringstream detail;
      detail.precision(2);
      detail << std::fixed << kLabels[index] << " c=" << detection.landmark_confidence[index];
      cv::putText(
        frame, detail.str(), detection.pixels[index] + cv::Point2f(6.0F, -6.0F),
        cv::FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv::LINE_AA);
      std::ostringstream coordinates;
      coordinates.precision(2);
      if (!confident) {
        coordinates << "low confidence";
      } else if (depth_valid) {
        coordinates << std::fixed << "(" << std::showpos << points[index].x << "," <<
          points[index].y << "," << std::noshowpos << points[index].z << ")m";
      } else {
        coordinates << "depth missing";
      }
      cv::putText(
        frame, coordinates.str(), detection.pixels[index] + cv::Point2f(6.0F, 10.0F),
        cv::FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv::LINE_AA);
    }
    cv::putText(
      frame, status, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55,
      valid ? cv::Scalar(70, 230, 100) : cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
  }

  void annotate_tracking_gap(cv::Mat & frame) const
  {
    if (tracked_box_) {
      cv::rectangle(frame, *tracked_box_, cv::Scalar(0, 180, 255), 2);
    }
    std::ostringstream status;
    status << "YOLO missed; holding controller " << missed_frames_ << "/" <<
      target_lock_max_missed_;
    cv::putText(
      frame, status.str(), cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX,
      0.55, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
  }

  void publish_debug_frame(cv::Mat frame, const std_msgs::msg::Header & header)
  {
    if (inference_fps_ > 0.0) {
      std::ostringstream rate;
      rate.precision(1);
      rate << std::fixed << "C++ YOLO " << inference_fps_ << " FPS | RGB-D dt " <<
        last_sync_delta_ms_ << " ms";
      cv::putText(
        frame, rate.str(), cv::Point(10, 52), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    if (publish_debug_) {
      debug_publisher_->publish(image_message(header, frame));
    }
    if (show_gui_) {
      cv::imshow("RealSense RGB-D pose (C++)", frame);
      cv::waitKey(1);
    }
  }

  void process(const FrameBundle & bundle)
  {
    cv::Mat frame = color_mat(*bundle.color);
    last_sync_delta_ms_ = bundle.sync_delta * 1000.0;
    if (bundle.sync_delta > sync_tolerance_) {
      ++sync_drop_count_;
      landmarks_publisher_->publish(point_cloud(bundle.color->header, std::nullopt, std::nullopt));
      std::ostringstream status;
      status << "color/depth mismatch " << std::lround(bundle.sync_delta * 1000.0) << " ms";
      cv::putText(
        frame, status.str(), cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX,
        0.55, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
      publish_debug_frame(std::move(frame), bundle.color->header);
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    auto detection = select_target(infer(frame));
    const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
    const double instant_fps = 1.0 / std::max(seconds, 1e-6);
    inference_fps_ = inference_fps_ <= 0.0 ? instant_fps :
      0.85 * inference_fps_ + 0.15 * instant_fps;
    if (!detection) {
      ++yolo_miss_count_;
      landmarks_publisher_->publish(point_cloud(bundle.color->header, std::nullopt, std::nullopt));
      if (tracked_box_ && missed_frames_ > 0) {
        annotate_tracking_gap(frame);
      } else {
        cv::putText(
          frame, "person not detected", cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX,
          0.55, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
      }
      publish_debug_frame(std::move(frame), bundle.color->header);
      return;
    }
    const auto points = reconstruct(*detection, *bundle.depth, *bundle.info);
    const bool left = side_ready(points, detection->landmark_confidence, kLeftRequired);
    const bool right = side_ready(points, detection->landmark_confidence, kRightRequired);
    landmarks_publisher_->publish(
      point_cloud(
        bundle.color->header, points, detection->landmark_confidence));
    std::string status;
    if (left && right) {
      ++complete_pose_count_;
      status = "C++ YOLO RGB-D pose ready";
    } else {
      ++partial_pose_count_;
      const auto left_missing = missing_required(
        points, detection->landmark_confidence, kLeftRequired);
      const auto right_missing = missing_required(
        points, detection->landmark_confidence, kRightRequired);
      if (left) {
        status = "L ready | R missing: " + right_missing;
      } else if (right) {
        status = "R ready | L missing: " + left_missing;
      } else {
        status = "L missing: " + left_missing + " | R missing: " + right_missing;
      }
    }
    if (publish_debug_ || show_gui_) {
      annotate(frame, *detection, points, status, left && right);
    }
    publish_debug_frame(std::move(frame), bundle.color->header);
  }

  void processing_loop()
  {
    while (running_.load() && rclcpp::ok()) {
      const auto bundle = take_frame();
      if (!bundle) {
        continue;
      }
      try {
        process(*bundle);
        const auto now = std::chrono::steady_clock::now();
        if (last_output_time_.time_since_epoch().count() != 0) {
          const double interval = std::chrono::duration<double>(now - last_output_time_).count();
          const double instant_output_fps = 1.0 / std::max(interval, 1e-6);
          output_fps_ = output_fps_ <= 0.0 ? instant_output_fps :
            0.85 * output_fps_ + 0.15 * instant_output_fps;
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 3000,
            "C++ pose rates: inference %.1f FPS, published %.1f FPS, RGB-D dt %.1f ms; "
            "drops sync=%llu yolo=%llu partial=%llu complete=%llu",
            inference_fps_, output_fps_, last_sync_delta_ms_,
            static_cast<unsigned long long>(sync_drop_count_),
            static_cast<unsigned long long>(yolo_miss_count_),
            static_cast<unsigned long long>(partial_pose_count_),
            static_cast<unsigned long long>(complete_pose_count_));
        }
        last_output_time_ = now;
      } catch (const c10::Error & error) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000, "C++ Torch inference failed: %s", error.what());
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000, "C++ RGB-D processing failed: %s", error.what());
      }
    }
  }

  std::string color_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string landmarks_topic_;
  std::string debug_topic_;
  std::string model_path_;
  int inference_size_;
  int torch_threads_;
  double person_confidence_;
  double min_landmark_confidence_;
  bool target_lock_enabled_;
  double target_lock_min_iou_;
  double target_lock_max_distance_;
  int target_lock_max_missed_;
  double keypoint_alpha_;
  double depth_scale_;
  int depth_radius_;
  int min_valid_depth_pixels_;
  double depth_cluster_tolerance_;
  double min_depth_;
  double max_depth_;
  double sync_tolerance_;
  double sync_wait_;
  bool publish_debug_;
  bool show_gui_;

  std::unique_ptr<torch::jit::script::Module> module_;
  std::atomic<bool> running_;
  std::thread worker_;
  std::mutex data_mutex_;
  std::condition_variable data_ready_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_color_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> depth_frames_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;
  std::size_t color_sequence_{0};
  std::size_t processed_color_sequence_{0};

  std::optional<cv::Rect2f> tracked_box_;
  std::optional<std::array<cv::Point2f, kLandmarkCount>> previous_pixels_;
  std::array<float, kLandmarkCount> previous_depths_;
  int missed_frames_{0};
  std::string tracking_state_{"acquired"};
  double inference_fps_{0.0};
  double output_fps_{0.0};
  double last_sync_delta_ms_{0.0};
  std::uint64_t sync_drop_count_{0};
  std::uint64_t yolo_miss_count_{0};
  std::uint64_t partial_pose_count_{0};
  std::uint64_t complete_pose_count_{0};
  std::chrono::steady_clock::time_point last_output_time_{};

  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr landmarks_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<DepthPoseDetectorCpp>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("depth_pose_detector_cpp"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
