#!/home/rosdemo/anaconda3/envs/yolo/bin/python3.10
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes



class YoloStereoDepth:
    def __init__(self):
        rospy.init_node("yolo_stereo_depth_node", anonymous=True)

        # 设置双目相机参数
        self.f = 700  # 焦距
        self.B = 0.1  # 基线距离（米）

        # 初始化StereoBM立体匹配对象和CvBridge
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.bridge = CvBridge()

        # 缓存图像和边界框消息
        self.left_image = None
        self.bounding_boxes = None
        # 订阅左目图像和边界框话题
        rospy.Subscriber("/left_cam/image_raw", Image, self.left_image_callback)
        rospy.Subscriber("/left/yolov8/BoundingBoxes", BoundingBoxes, self.bounding_boxes_callback)

    def left_image_callback(self, msg):
        """接收左目图像的回调函数"""
        rospy.loginfo("start!")
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo("收到左目图像消息")
            self.process_data_if_ready()
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")

    def bounding_boxes_callback(self, msg):
        """接收边界框信息的回调函数"""
        self.bounding_boxes = msg
        rospy.loginfo("收到YOLO边界框消息")
        self.process_data_if_ready()

    def process_data_if_ready(self):
        """当图像和边界框消息都准备好时，进行处理"""
        if self.left_image is None or self.bounding_boxes is None:
            return

        try:
            # 读取右目图像
            right_img_msg = rospy.wait_for_message("/right_cam/image_raw", Image, timeout=5.0)
            right_image = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
            right_bounding_boxes_sub = rospy.wait_for_message("/right/yolov8/BoundingBoxes", BoundingBoxes, timeout=5.0)

            # 遍历每个检测到的边界框
            for left_box in self.bounding_boxes.bounding_boxes:
                x_left = left_box.xmin
                y_left = left_box.ymin
                w_left = left_box.xmax - left_box.xmin
                h_left = left_box.ymax - left_box.ymin
                detector_name_left = left_box.Class
                rospy.loginfo("开始匹配！")
                rospy.loginfo(f"类别:{detector_name_left}")
                for right_box in right_bounding_boxes_sub.bounding_boxes:
                    if detector_name_left == right_box.Class:
                        # 从左右图像中裁剪目标区域
                        rospy.loginfo("开始计算")
                        left_cropped = self.left_image[left_box.ymin:left_box.ymax, left_box.xmin:left_box.xmax]
                        right_cropped = right_image[right_box.ymin:right_box.ymax, right_box.xmin:right_box.xmax]

                        # 计算视差图
                        disparity_map = self.stereo.compute(cv2.cvtColor(left_cropped, cv2.COLOR_BGR2GRAY),
                                                            cv2.cvtColor(right_cropped, cv2.COLOR_BGR2GRAY))

                        # 计算平均视差，去除无效的视差值
                        disparity = np.mean(disparity_map[disparity_map > 0])

                        if disparity > 0:
                            # 使用视差计算深度
                            Z = self.f * self.B / disparity
                            rospy.loginfo(f"目标类别: {detector_name_left}, 距离: {Z:.2f} 米")

            # 清空缓存，等待下一组数据
            self.left_image = None
            self.bounding_boxes = None

        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")
        except rospy.ROSException as e:
            rospy.logerr("获取右目图像超时")

if __name__ == '__main__':
    try:
        YoloStereoDepth()
        # rospy.INFO("111")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
