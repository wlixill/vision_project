## 文件架构

```
——————build  编译文件
——————devel  开发文件
——————src——————camera_spilt 相机分割节点功能包
        ——————YOLOv8        YOLOv8源码功能包
        ——————Yolov8_ros       
            ——————yolov8_ros    启动YOLOv8以及测距功能包
            ——————yolov8_ros_msgs   自定义消息功能包
——————CMakeLists.txt
——————README.md           
```
## 环境配置
```
本项目基于ubuntu20.04 ros-noetic anaconda的环境下启动
```
```bash
    conda create -n yolo python=3.10
    conda activate yolo
    cd cv_ws/src/YOLOv8
    pip install -r requirements.txt
```
### pytorch 
    gpu 可自行前往官网查询

    cpu 可自行前往官网查询

## 项目启动
本次双目相机为直驱，因此需要依赖于ros-noetic-usb_cam,并且只有特定的分辨率才能打开两个摄像头

```bash
    sudo apt install ros-noetic-usb_cam*
    roscd usb_cam
    sudo gedit launch/usb_cam-test.launch
    ```
        <param name="image_width" value="1280" />
        <param name="image_height" value="480" />
    ```
    cd 
    source /opt/ros/noetic/setup.bash
    roslaunch usb_cam usb_cam-test.launch
```

```bash
    cd cv_ws
    catkin_make
    source devel/setup.bash
    roslaunch camera_spilt camera_spilt_calibration.launch
```

```bash
    cd cv_ws
    conda activate yolo
    source devel/setup.bash
    roslaunch yolov8_ros yolo_v8_left.launch  
    && roslaunch yolov8_ros yolo_v8_right.launch     
```   
```bash
    cd cv_ws
    source devel/setup.bash
    roslaunch yolov8_ros yolo_stero.launch
```


# TODO
1. 无法通过launch传入参数(Fine)
2. 测距不准
3. yolov8权重文件需要更换(Fine)
4. 算法需要优化,匹配计算的物体错误
5. 把解算出来的数据put到图像上(Fine)
6. 把launch文件集成进一个文件里面（没必要）