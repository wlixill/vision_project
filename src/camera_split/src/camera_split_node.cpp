#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <camera_info_manager/camera_info_manager.h>

#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

using namespace std;

class CameraSplitter
{
public:
    CameraSplitter():nh_("~"),it_(nh_)
    {
        image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &CameraSplitter::imageCallback, this);
        image_pub_left_ = it_.advertiseCamera("/left_cam/image_raw", 1);
        image_pub_right_ = it_.advertiseCamera("/right_cam/image_raw", 1);
        cinfo_ = boost::shared_ptr<camera_info_manager::CameraInfoManager>(new camera_info_manager::CameraInfoManager(nh_));

        //读取参数服务器参数，得到左右相机参数文件的位置
        string left_cal_file = nh_.param<std::string>("left_cam_file", "");
        string right_cal_file = nh_.param<std::string>("right_cam_file", "");
        if(!left_cal_file.empty())
        {
            if(cinfo_->validateURL(left_cal_file)) {
                cout<<"Load left camera info file: "<<left_cal_file<<endl;
                cinfo_->loadCameraInfo(left_cal_file);
                ci_left_ = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo(cinfo_->getCameraInfo()));
            }
            else {
                cout<<"Can't load left camera info file: "<<left_cal_file<<endl;
                ros::shutdown();
            }
        }
        else {
            cout<<"Did not specify left camera info file." <<endl;
            ci_left_=sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo());
        }
        if(!right_cal_file.empty())
        {
            if(cinfo_->validateURL(right_cal_file)) {
                cout<<"Load right camera info file: "<<right_cal_file<<endl;
                cinfo_->loadCameraInfo(right_cal_file);
                ci_right_ = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo(cinfo_->getCameraInfo()));
            }
            else {
                cout<<"Can't load right camera info file: "<<right_cal_file<<endl;
                ros::shutdown();
            }
        }
        else {
            cout<<"Did not specify right camera info file." <<endl;
            ci_right_=sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo());
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        namespace enc= sensor_msgs::image_encodings;
        cv_ptr= cv_bridge::toCvShare(msg, enc::BGR8);
        //截取ROI(Region Of Interest)，即左右图像，会将原图像数据拷贝出来。
        leftImgROI_=cv_ptr->image(cv::Rect(0,0,cv_ptr->image.cols/2, cv_ptr->image.rows));
        rightImgROI_=cv_ptr->image(cv::Rect(cv_ptr->image.cols/2,0, cv_ptr->image.cols/2, cv_ptr->image.rows ));
        //创建两个CvImage, 用于存放原始图像的左右部分。CvImage创建时是对Mat进行引用的,不会进行数据拷贝
        leftImgPtr_=cv_bridge::CvImagePtr(new cv_bridge::CvImage(cv_ptr->header, cv_ptr->encoding,leftImgROI_) );
        rightImgPtr_=cv_bridge::CvImagePtr(new cv_bridge::CvImage(cv_ptr->header, cv_ptr->encoding,rightImgROI_) );

        //发布到/left_cam/image_raw和/right_cam/image_raw
        ci_left_->header = cv_ptr->header; 	//很重要，不然会提示不同步导致无法去畸变
        ci_right_->header = cv_ptr->header;
        sensor_msgs::ImagePtr leftPtr =leftImgPtr_->toImageMsg();
        sensor_msgs::ImagePtr rightPtr =rightImgPtr_->toImageMsg();
        leftPtr->header=msg->header; 		//很重要，不然输出的图象没有时间戳
        rightPtr->header=msg->header;
        image_pub_left_.publish(leftPtr,ci_left_);
        image_pub_right_.publish(rightPtr,ci_right_);
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::CameraPublisher image_pub_left_;
    image_transport::CameraPublisher image_pub_right_;
    boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_;
    sensor_msgs::CameraInfoPtr ci_left_;
    sensor_msgs::CameraInfoPtr ci_right_;

    cv::Mat leftImgROI_;
    cv::Mat rightImgROI_;
    cv_bridge::CvImagePtr leftImgPtr_=NULL;
    cv_bridge::CvImagePtr rightImgPtr_=NULL;
    
};

int main(int argc,char** argv)
{
    ros::init(argc, argv, "camera_split");
    CameraSplitter cameraSplitter;
    ros::spin();
    return 0;
}

