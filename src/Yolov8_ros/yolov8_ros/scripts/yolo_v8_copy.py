#!/home/rosdemo/anaconda3/envs/yolo/bin/python3.10
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
from ultralytics import YOLO
from time import time

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes


class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        left_image_topic = rospy.get_param(
            '~right_image_topic', '/camera/color/image_raw')
        right_image_topic = rospy.get_param(
            '~left_image_topic', '/camera/color/image_raw')
        left_pub_topic = rospy.get_param('~left_pub_topic', '/yolov8/BoundingBoxes')
        right_pub_topic = rospy.get_param('~right_pub_topic', '/yolov8/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.visualize = rospy.get_param('~visualize', 'True')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self.model = YOLO(weight_path)
        self.model.fuse()

        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(left_image_topic, Image, self.left_image_callback,
                                          queue_size=1, buff_size=52428800)

        # output publishers
        self.position_pub = rospy.Publisher(
            left_pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/left/yolov8/detection_image',  Image, queue_size=1)

        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for left image.")
            rospy.sleep(2)

    def left_image_callback(self, image):

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image, show=False, conf=0.3)

        self.left_dectshow(results, image.height, image.width)

        cv2.waitKey(3)

    def left_dectshow(self, results, height, width):

        self.frame = results[0].plot()
        print(str(results[0].speed['inference']))
        fps = 1000.0/ results[0].speed['inference']
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        for result in results[0].boxes:
            boundingBox = BoundingBox()
            boundingBox.xmin = np.int64(result.xyxy[0][0].item())
            boundingBox.ymin = np.int64(result.xyxy[0][1].item())
            boundingBox.xmax = np.int64(result.xyxy[0][2].item())
            boundingBox.ymax = np.int64(result.xyxy[0][3].item())
            boundingBox.Class = results[0].names[result.cls.item()]
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)
        self.position_pub.publish(self.boundingBoxes)
        self.left_publish_image(self.frame, height, width)

        if self.visualize :
            cv2.imshow('left', self.frame)

    def left_publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)


        self.color_sub = rospy.Subscriber(right_image_topic, Image, self.right_image_callback,
                                          queue_size=1, buff_size=52428800)

        # output publishers
        self.position_pub = rospy.Publisher(
            right_pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/right/yolov8/detection_image',  Image, queue_size=1)

        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for right image.")
            rospy.sleep(2)

    def right_image_callback(self, image):

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image, show=False, conf=0.3)

        self.dectshow(results, image.height, image.width)

        cv2.waitKey(3)

    def dectshow(self, results, height, width):

        self.frame = results[0].plot()
        print(str(results[0].speed['inference']))
        fps = 1000.0/ results[0].speed['inference']
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        for result in results[0].boxes:
            boundingBox = BoundingBox()
            boundingBox.xmin = np.int64(result.xyxy[0][0].item())
            boundingBox.ymin = np.int64(result.xyxy[0][1].item())
            boundingBox.xmax = np.int64(result.xyxy[0][2].item())
            boundingBox.ymax = np.int64(result.xyxy[0][3].item())
            boundingBox.Class = results[0].names[result.cls.item()]
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)
        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(self.frame, height, width)

        if self.visualize :
            cv2.imshow('right', self.frame)

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)


def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()

