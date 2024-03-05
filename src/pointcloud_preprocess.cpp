#include "ig_lio/pointcloud_preprocess.h"
#include "ig_lio/timer.h"


extern Timer timer;

float pointDistance(PointType p)
{
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

void getColor(float p, float np, float &r, float &g, float &b)
{
  float inc = 6.0 / np;
  float x = p * inc;
  r = 0.0f;
  g = 0.0f;
  b = 0.0f;
  if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
    r = 1.0f;
  else if (4 <= x && x <= 5)
    r = x - 4;
  else if (1 <= x && x <= 2)
    r = 1.0f - (x - 1);

  if (1 <= x && x <= 3)
    g = 1.0f;
  else if (0 <= x && x <= 1)
    g = x - 0;
  else if (3 <= x && x <= 4)
    g = 1.0f - (x - 3);

  if (3 <= x && x <= 5)
    b = 1.0f;
  else if (2 <= x && x <= 3)
    b = x - 2;
  else if (5 <= x && x <= 6)
    b = 1.0f - (x - 5);
  r *= 255.0;
  g *= 255.0;
  b *= 255.0;
}

void PointCloudPreprocess::Process(
    const livox_ros_driver2::CustomMsg::ConstPtr& msg,
    pcl::PointCloud<PointType>::Ptr& cloud_out,
    cv::Mat image_in,cv::Mat image_out, cv::Mat intrisicMat,
    Eigen::Affine3f exT,const double last_start_time) {
  double time_offset =
      (msg->header.stamp.toSec() - last_start_time) * 1000.0;  // ms

  //1.
    Eigen::Vector3f livox_pt;
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

#pragma omp parallel for num_threads(omp_get_max_threads())
  for (size_t i = 0; i < msg->point_num; ++i) {
    // 对数据进行筛选
    if ((msg->points[i].line < num_scans_) && (i % config_.point_filter_num == 0) &&
       ((msg->points[i].tag & 0x30) == 0x10 ||
       (msg->points[i].tag & 0x30) == 0x00) &&
        !HasInf(msg->points[i]) && !HasNan(msg->points[i]) &&
        !IsNear(msg->points[i], msg->points[i - 1]))
     {
        //2.转换到相机系
        Eigen::Vector3f pt(msg->points[i].x, msg->points[i].y, msg->points[i].z);
        livox_pt = exT * pt;
      
        X.at<double>(0,0) = livox_pt(0);
        X.at<double>(1,0) = livox_pt(1);
        X.at<double>(2,0) = livox_pt(2);
        X.at<double>(3,0) = 1;

        //3.转换到像素坐标系
        Y = intrisicMat  * X; 
        cv::Point2f u_v;
        u_v.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        u_v.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        if(u_v.x < 0 || u_v.y < 0 || u_v.x > image_in.cols || u_v.y > image_in.rows)
                continue;
      #pragma omp critical
      {
          PointType point;
          PointXYZIRGBNormal point_i;
          point_i.normal_x = point.normal_x = 0;
          point_i.normal_x = point.normal_y = 0;
          point_i.normal_x = point.normal_z = 0;
          point_i.x = point.x = msg->points[i].x;
          point_i.y = point.y = msg->points[i].y;
          point_i.z = point.z = msg->points[i].z;
          point_i.r = point.r = image_in.at<cv::Vec3b>(u_v.y,u_v.x)[0]; //(row,col)
          point_i.g = point.g = image_in.at<cv::Vec3b>(u_v.y,u_v.x)[1];
          point_i.b = point.b = image_in.at<cv::Vec3b>(u_v.y,u_v.x)[2];
          point_i.intensity = msg->points[i].reflectivity;
          point_i.curvature = point.curvature = time_offset + msg->points[i].offset_time * 1e-6;  // ms
          cloud_out->push_back(point);

          //4.融合图片
          float dist,r,g,b;
          dist = pointDistance(point);
          getColor(dist,50,r,g,b);
          cv::circle(image_out, cv::Point2f(u_v.x,u_v.y), 0, cv::Scalar(r, g, b),5);
      }
    }else{
      continue;
    }
  }
  //5.雷达视角
//  pcl::transformPointCloud(*cloud_out,*cloud_out,exT);
 std::cout << ">>>> Color points nums: " << cloud_out->points.size() << std::endl;
}

void PointCloudPreprocess::Process(
    const livox_ros_driver2::CustomMsg::ConstPtr& msg,
    pcl::PointCloud<PointType>::Ptr& cloud_out,
    const double last_start_time) {
  double time_offset =
      (msg->header.stamp.toSec() - last_start_time) * 1000.0;  // ms

  for (size_t i = 1; i < msg->point_num; ++i) {
    if ((msg->points[i].line < num_scans_) &&
        ((msg->points[i].tag & 0x30) == 0x10 ||
         (msg->points[i].tag & 0x30) == 0x00) &&
        !HasInf(msg->points[i]) && !HasNan(msg->points[i]) &&
        !IsNear(msg->points[i], msg->points[i - 1]) &&
        (i % config_.point_filter_num == 0)) {
      PointType point;
      point.normal_x = 0;
      point.normal_y = 0;
      point.normal_z = 0;
      point.x = msg->points[i].x;
      point.y = msg->points[i].y;
      point.z = msg->points[i].z;
      float dist,r,g,b;
      dist = pointDistance(point);
      getColor(dist,50,r,g,b);
      point.r = r;
      point.g = g;
      point.b = b;
      // point.intensity = msg->points[i].reflectivity;
      point.curvature = time_offset + msg->points[i].offset_time * 1e-6;  // ms
      cloud_out->push_back(point);
    }
  }
  std::cout << ">>>> Points nums: " << cloud_out->points.size() << std::endl;
}

void PointCloudPreprocess::Process(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    pcl::PointCloud<PointType>::Ptr& cloud_out) {
  switch (config_.lidar_type) {
  case LidarType::VELODYNE:
    ProcessVelodyne(msg, cloud_out);
    break;
  case LidarType::OUSTER:
    ProcessOuster(msg, cloud_out);
    break;
  default:
    LOG(INFO) << "Error LiDAR Type!!!" << std::endl;
    exit(0);
  }
}

void PointCloudPreprocess::ProcessVelodyne(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    pcl::PointCloud<PointType>::Ptr& cloud_out) {
  pcl::PointCloud<VelodynePointXYZIRT> cloud_origin;
  pcl::fromROSMsg(*msg, cloud_origin);

  // These variables only works when no point timestamps given
  int plsize = cloud_origin.size();
  double omega_l = 3.61;  // scan angular velocity
  std::vector<bool> is_first(num_scans_, true);
  std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
  std::vector<float> time_last(num_scans_, 0.0);  // last offset time
  if (cloud_origin.back().time > 0) {
    has_time_ = true;
  } else {
    LOG(INFO) << "origin cloud has not timestamp";
    has_time_ = false;
    double yaw_first =
        atan2(cloud_origin.points[0].y, cloud_origin.points[0].x) * 57.29578;
    double yaw_end = yaw_first;
    int layer_first = cloud_origin.points[0].ring;
    for (uint i = plsize - 1; i > 0; i--) {
      if (cloud_origin.points[i].ring == layer_first) {
        yaw_end = atan2(cloud_origin.points[i].y, cloud_origin.points[i].x) *
                  57.29578;
        break;
      }
    }
  }

  cloud_out->reserve(cloud_origin.size());

  for (size_t i = 0; i < cloud_origin.size(); ++i) {
    if ((i % config_.point_filter_num == 0) && !HasInf(cloud_origin.at(i)) &&
        !HasNan(cloud_origin.at(i))) {
      PointType point;
      point.normal_x = 0;
      point.normal_y = 0;
      point.normal_z = 0;
      point.x = cloud_origin.at(i).x;
      point.y = cloud_origin.at(i).y;
      point.z = cloud_origin.at(i).z;
      point.r = 255;
      point.g = 0;
      point.b = 0;
      // point.intensity = cloud_origin.at(i).intensity;
      if (has_time_) {
        // curvature unit: ms
        point.curvature = cloud_origin.at(i).time * config_.time_scale;
        // std::cout<<point.curvature<<std::endl;
        // if(point.curvature < 0){
        //     std::cout<<"time < 0 : "<<point.curvature<<std::endl;
        // }
      } else {
        int layer = cloud_origin.points[i].ring;
        double yaw_angle = atan2(point.y, point.x) * 57.2957;

        if (is_first[layer]) {
          yaw_fp[layer] = yaw_angle;
          is_first[layer] = false;
          point.curvature = 0.0;
          yaw_last[layer] = yaw_angle;
          time_last[layer] = point.curvature;
          continue;
        }

        // compute offset time
        if (yaw_angle <= yaw_fp[layer]) {
          point.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        } else {
          point.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (point.curvature < time_last[layer])
          point.curvature += 360.0 / omega_l;

        if (!std::isfinite(point.curvature)) {
          continue;
        }

        yaw_last[layer] = yaw_angle;
        time_last[layer] = point.curvature;
      }

      cloud_out->push_back(point);
    }
  }
}

void PointCloudPreprocess::ProcessOuster(
    const sensor_msgs::PointCloud2::ConstPtr& msg,
    pcl::PointCloud<PointType>::Ptr& cloud_out) {
  pcl::PointCloud<OusterPointXYZIRT> cloud_origin;
  pcl::fromROSMsg(*msg, cloud_origin);

  for (size_t i = 0; i < cloud_origin.size(); ++i) {
    if ((i % config_.point_filter_num == 0) && !HasInf(cloud_origin.at(i)) &&
        !HasNan(cloud_origin.at(i))) {
      PointType point;
      point.normal_x = 0;
      point.normal_y = 0;
      point.normal_z = 0;
      point.x = cloud_origin.at(i).x;
      point.y = cloud_origin.at(i).y;
      point.z = cloud_origin.at(i).z;
      point.r = 255;
      point.g = 0;
      point.b = 0;
      // point.intensity = cloud_origin.at(i).intensity;
      point.curvature = cloud_origin.at(i).t * 1e-6; //ms
      cloud_out->push_back(point);
    }
  }
}

template <typename T>
inline bool PointCloudPreprocess::HasInf(const T& p) {
  return (std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z));
}

template <typename T>
inline bool PointCloudPreprocess::HasNan(const T& p) {
  return (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z));
}

template <typename T>
inline bool PointCloudPreprocess::IsNear(const T& p1, const T& p2) {
  return ((abs(p1.x - p2.x) < 1e-7) || (abs(p1.y - p2.y) < 1e-7) ||
          (abs(p1.z - p2.z) < 1e-7));
}
