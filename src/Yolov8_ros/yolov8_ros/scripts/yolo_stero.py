#!/home/rosdemo/anaconda3/envs/yolo/bin/python3.10
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
import sys

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from message_filters import ApproximateTimeSynchronizer, Subscriber,  TimeSynchronizer


left_detection_image_topic = rospy.get_param(
'~left_detection_image_topic', '/left/yolov8/detection_image')
# 左识别框消息
left_BoundingBoxes_topic = rospy.get_param(
'~left_BoundingBoxes_topic', 'left_BoundingBoxes_topic')
# 右识别图
right_detection_image_topic = rospy.get_param(
'~right_detection_image_topic', '/right/yolov8/detection_image')
# 右边识别框消息
right_BoundingBoxes_topic = rospy.get_param(
'~right_BoundingBoxes_topic', 'right_BoundingBoxes_topic')

#  步骤
#  接受识别后的图片
#  根据检测到的识别框，剪裁roi，然后放入StereoBM算出视差dis
#  最后根据公式z = f*b算出距离z

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# bridge = CvBridge()

# f =  焦距 x 分辨率 / 传感器宽度
# 焦距为3.5mm
# 分辨率单个为640像素
# 传感器宽度 12mm

f = 187 # 像素单位
b = 0.12 # 基线即两个相机之间的距离单位:m
def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


def callback(left_img_msg, bounding_boxes_msg):
    try:
        # 将ROS图像消息转换为Opencv图像
        # left_image = bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
        left_image = imgmsg_to_cv2(left_img_msg)
        
        right_image = rospy.wait_for_message("/right/yolov8/detection_image", Image)
        # right_image = bridge.imgmsg_to_cv2(right_image, "bgr8")
        right_image = imgmsg_to_cv2(right_image)

        right_bounding_boxes_sub = rospy.wait_for_message("/right/yolov8/BoundingBoxes", BoundingBoxes)
        

        # left_img_pub = rospy.Publisher("process_left_image", Image, queue_size=1)
        # right_img_pub = rospy.Publisher("process_right_image", Image, queue_size=1)
        process_left_image = np.copy(left_image)
        process_right_image = np.copy(right_image)

        for left_box in bounding_boxes_msg.bounding_boxes:
            detector_name_left = left_box.Class
            for right_box in right_bounding_boxes_sub.bounding_boxes:
                if detector_name_left == right_box.Class:
                    rospy.loginfo("开始计算")
                    left_cropped = process_left_image[left_box.ymin:left_box.ymax, left_box.xmin:left_box.xmax]
                    right_cropped = process_right_image[right_box.ymin:right_box.ymax, right_box.xmin:right_box.xmax]
                    if left_cropped.shape != right_cropped.shape:
                        right_cropped = cv2.resize(right_cropped, (left_cropped.shape[1], left_cropped.shape[0]))
                    disparity_map = stereo.compute(cv2.cvtColor(left_cropped, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_cropped, cv2.COLOR_BGR2GRAY))
                     # 计算目标的平均视差，去除无效的视差值
                    valid_disparity = np.mean(disparity_map[disparity_map > 0])
                    if valid_disparity.size > 0:
                        disparity = np.median(valid_disparity)
                        if disparity > 0:
                            z = f * b / disparity
                            rospy.loginfo(f"目标类别:{detector_name_left}, 距离:{z:.2f}米")
                            label_text = f"{detector_name_left}:{z:2f}m"


                            
                            cv2.putText(process_left_image, label_text, (left_box.xmin, left_box.ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(process_right_image, label_text, (left_box.xmin, left_box.ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("process_left_image", process_left_image)
            cv2.imshow("process_right_image", process_right_image)
            cv2.waitKey(1)
        

    except CvBridgeError as e:
        rospy.logerr(f"图像转换错误: {e}")


def main():
    rospy.init_node('yolo_stero', anonymous=True)
    left_image_sub = Subscriber("/left/yolov8/detection_image", Image)
    left_bounding_boxes_sub = Subscriber("/left/yolov8/BoundingBoxes", BoundingBoxes)
    ats = ApproximateTimeSynchronizer([left_image_sub, left_bounding_boxes_sub], queue_size=10, slop=5)
    ats.registerCallback(callback)
    rospy.loginfo("测距节点启动")
    rospy.spin()


if __name__ == "__main__":

    main()
