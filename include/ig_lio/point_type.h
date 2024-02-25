/*
 * @Description: define point type
 * @Autor: Zijie Chen
 * @Date: 2023-12-25 22:28:00
 */

#ifndef POINT_TYPE_H_
#define POINT_TYPE_H_

#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

             
// struct PointXYZIRGBNormal
// {
//     PCL_ADD_POINT4D;
//     PCL_ADD_INTENSITY;
//     PCL_ADD_RGB;
    
//     float curvature;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGBNormal,
//                                     (float,x,x)
//                                     (float,y,y)
//                                     (float,z,z)
//                                     (float ,intensity,intensity)
//                                     (uint_32t,rgb,rgb)
//                                     (float,curvature,curvature))

// using PointType = PointXYZIRGBNormal;
using PointType = pcl::PointXYZRGBNormal;
using CloudType = pcl::PointCloud<PointType>;
using CloudPtr = CloudType::Ptr;

struct PointWithCovariance {
  PCL_ADD_POINT4D
  int idx;
  float cov[6];
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointWithCovariance,
    (float, x, x)(float, y, y)(float, z, z)(int, idx, idx)(float[6], cov, cov))

using PointCovType = PointWithCovariance;
using CloudCovType = pcl::PointCloud<PointWithCovariance>;
using CloudCovPtr = CloudCovType::Ptr;

#endif