# click_distance.py
import pyrealsense2 as rs
import numpy as np
import cv2

# 启动相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# 深度对齐到彩色图
align = rs.align(rs.stream.color)

# 鼠标点击回调
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = param["depth"]
        dist = depth.get_distance(x, y)  # 单位：米
        print(f"点击位置 ({x}, {y})  距离: {dist*100:.1f} cm")
        param["click"] = (x, y, dist)

state = {}
cv2.namedWindow("RealSense")
cv2.setMouseCallback("RealSense", on_click, state)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        state["depth"] = depth_frame
        color_img = np.asanyarray(color_frame.get_data())

        # 显示点击点和距离
        if "click" in state:
            x, y, d = state["click"]
            cv2.circle(color_img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(color_img, f"{d*100:.1f}cm",
                        (x+10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("RealSense", color_img)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()