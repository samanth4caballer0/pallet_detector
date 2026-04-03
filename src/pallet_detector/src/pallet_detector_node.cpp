#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> //instead of passthrough for 3D ROI
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/centroid.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

class PalletDetector
{
public:
  using PointT = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<PointT>;

  PalletDetector(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh), pnh_(pnh)
  {
    // Parameters
    pnh_.param<std::string>("cloud_topic", cloud_topic_, "/demo/rgbd_camera/depth/points");
    pnh_.param<std::string>("output_frame", output_frame_, ""); // leave empty to use incoming frame

    // ROI
    pnh_.param("x_min", x_min_, -1.2);
    pnh_.param("x_max", x_max_, 1.2);
    pnh_.param("y_min", y_min_, 0.2);
    pnh_.param("y_max", y_max_, 1.0);
    pnh_.param("z_min", z_min_, 1.0);
    pnh_.param("z_max", z_max_, 3.5);

    // Filtering
    pnh_.param("voxel_leaf", voxel_leaf_, 0.02); // meters
    pnh_.param("use_sor", use_sor_, false);
    pnh_.param("sor_mean_k", sor_mean_k_, 30);
    pnh_.param("sor_stddev", sor_stddev_, 1.0);

    // Plane removal
    pnh_.param("plane_dist_thresh", plane_dist_thresh_, 0.02);
    pnh_.param("plane_max_iters", plane_max_iters_, 100);

    // Clustering
    pnh_.param("cluster_tolerance", cluster_tolerance_, 0.05);
    pnh_.param("cluster_min_size", cluster_min_size_, 500);
    pnh_.param("cluster_max_size", cluster_max_size_, 200000);

    // Pallet expected dims (meters)
    pnh_.param("pallet_length", pallet_L_, 1.2);
    pnh_.param("pallet_width", pallet_W_, 0.8);
    pnh_.param("pallet_height", pallet_H_, 0.15);

    pnh_.param("tol_length", tol_L_, 0.10);
    pnh_.param("tol_width", tol_W_, 0.10);
    pnh_.param("tol_height", tol_H_, 0.06);

    // Top-face occupancy scoring params (used as one cue, not the only cue)
    // Idea: pallets are not fully solid on top due to deck-board gaps.
    pnh_.param("occupancy_cell_size", occupancy_cell_size_, 0.02);    // top grid cell size (m)
    pnh_.param("min_fill_ratio", min_fill_ratio_, 0.3);               // nominal pallet top fill lower bound
    pnh_.param("max_fill_ratio", max_fill_ratio_, 0.8);               // nominal pallet top fill upper bound
    pnh_.param("enable_top_fill_gate", enable_top_fill_gate_, false); // optional hard gate (instead of only soft score)

    // Front/side structural scoring params (template-like fork-pocket cues)
    // Idea: project to a vertical face and look for lower-half void columns.
    pnh_.param("front_cell_size", front_cell_size_, 0.02); // vertical face grid cell size (m)
    pnh_.param("front_fill_min", front_fill_min_, 0.12);
    pnh_.param("front_fill_max", front_fill_max_, 0.85);
    pnh_.param("front_lower_height_ratio", front_lower_height_ratio_, 0.55); // lower region where pockets are expected
    pnh_.param("front_void_column_fill_max", front_void_column_fill_max_, 0.25);
    pnh_.param("front_min_void_columns", front_min_void_columns_, 2);
    pnh_.param("front_upper_fill_min", front_upper_fill_min_, 0.25);

    // Height profile scoring params (bimodality of deck layers)
    // Idea: pallet tends to have two main strata (top deck + lower structure).
    pnh_.param("height_bins", height_bins_, 12);
    pnh_.param("height_valley_ratio_max", height_valley_ratio_max_, 0.45);

    // Weighted fusion params: combine dimension + structure cues into one confidence score.
    pnh_.param("w_dim", w_dim_, 0.15);
    pnh_.param("w_top", w_top_, 0.25);
    pnh_.param("w_front", w_front_, 0.45);
    pnh_.param("w_height", w_height_, 0.15);
    pnh_.param("min_front_score", min_front_score_, 0.35); // keep strong structural discrimination against solid boxes
    pnh_.param("min_total_score", min_total_score_, 0.62); // final acceptance threshold per cluster

    // Temporal tracking params — smooth pose and add confidence hysteresis.
    // Hysteresis helps avoid flip-flopping when detections are noisy or briefly occluded.
    pnh_.param("track_gate_dist", track_gate_dist_, 0.3);        // max allowed jump (m) for same object
    pnh_.param("track_max_miss", track_max_miss_, 5);            // frames to coast on last pose
    pnh_.param("track_alpha", track_alpha_, 0.3);                // pose smoothing factor
    pnh_.param("track_conf_alpha", track_conf_alpha_, 0.2);      // temporal smoothing on score confidence
    pnh_.param("track_acquire_conf", track_acquire_conf_, 0.70); // confidence to start a fresh track
    pnh_.param("track_keep_conf", track_keep_conf_, 0.45);       // lower threshold to keep existing track alive
    pnh_.param("track_acquire_hits", track_acquire_hits_, 2);    // consecutive high-confidence hits to acquire

    sub_ = nh_.subscribe(cloud_topic_, 1, &PalletDetector::cloudCb, this);

    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pallet_pose", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("pallet_marker", 1);
    roi_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_roi", 1);
    no_plane_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_no_plane", 1);

    ROS_INFO_STREAM("pallet_detector: subscribing to " << cloud_topic_);
  }

private: // MAIN CALLBACK
  void cloudCb(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    CloudT::Ptr cloud(new CloudT); // input cloud is received
    pcl::fromROSMsg(*msg, *cloud); // convert from ROS message to PCL point cloud
    if (cloud->empty())            // safety check
      return;

    const std::string frame = output_frame_.empty() ? msg->header.frame_id : output_frame_; // use incoming frame if output_frame is empty

    // 1) ROI crop box
    CloudT::Ptr roi(new CloudT); // output cloud after ROI crop
    roi = cropBoxROI(cloud);

    /////////////////////publish ROI cloud for debug visualisation//////////
    if (roi_pub_.getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2 roi_msg;
      pcl::toROSMsg(*roi, roi_msg);
      roi_msg.header = msg->header;
      roi_pub_.publish(roi_msg);
    }
    ////////////////////////////////////////////////////////////////////////

    // 2) Voxel downsample
    CloudT::Ptr ds(new CloudT); // output cloud after downsampling
    pcl::VoxelGrid<PointT> vg;  // voxel grid filter for downsampling
    vg.setInputCloud(roi);      // vg is PCL's built-in voxel grid filter for downsampling point clouds by averaging points within a 3D grid of specified leaf size (voxel_leaf_)
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.filter(*ds);
    if (ds->size() < 100) // if too few points remain after downsampling, skip processing
      return;

    // 3) Optional outlier removal
    CloudT::Ptr filtered(new CloudT);
    if (use_sor_)
    {
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(ds);
      sor.setMeanK(sor_mean_k_);
      sor.setStddevMulThresh(sor_stddev_);
      sor.filter(*filtered);
    }
    else
    {
      filtered = ds;
    }

    // 4) Plane segmentation (remove floor/table)
    CloudT::Ptr no_plane(new CloudT);
    removeDominantPlane(filtered, no_plane); // applied function (helper defined below)
    if (no_plane->size() < 200)              // if too few points remain after plane removal, skip processing (like if entire cloud is just floor)
      return;

    // publish no-plane cloud for debug visualisation (floor removed)
    if (no_plane_pub_.getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2 np_msg;
      pcl::toROSMsg(*no_plane, np_msg);
      np_msg.header = msg->header;
      no_plane_pub_.publish(np_msg);
    }

    // 5) Euclidean clustering
    std::vector<pcl::PointIndices> cluster_indices; // output cluster indices (vector of clusters, each cluster is vector of point indices)
    euclideanClusters(no_plane, cluster_indices);

    ROS_INFO_STREAM_THROTTLE(1.0, "cloud=" << cloud->size()
                                           << " roi=" << roi->size()
                                           << " ds=" << ds->size()
                                           << " filtered=" << filtered->size()
                                           << " no_plane=" << no_plane->size()
                                           << " clusters detected = " << cluster_indices.size());

    // 6) Score clusters by geometry + structure, then pick best candidate.
    bool candidate_found = false;           // whether at least one cluster passed all gates
    geometry_msgs::PoseStamped best_pose;   // pose of current best cluster
    visualization_msgs::Marker best_marker; // marker of current best cluster
    double best_total_score = -1.0;         // higher is better in this pipeline
    double best_dim_score = 0.0;            // for logging/debug only
    double best_top_score = 0.0;            // for logging/debug only
    double best_front_score = 0.0;          // for logging/debug only
    double best_height_score = 0.0;         // for logging/debug only
    double best_top_fill_ratio = 0.0;       // for logging/debug only
    double best_front_fill_ratio = 0.0;     // for logging/debug only
    int best_front_void_cols = 0;           // for logging/debug only

    int cluster_id = 0;
    for (const auto &idx : cluster_indices) // FOR EACH CLUSTER -> idx.indices is vector of point indices for this cluster
    {                                       // translating index numbers into real 3D points, one cluster at a time
      CloudT::Ptr cluster(new CloudT);      // point cloud for this cluster
      cluster->reserve(idx.indices.size());
      for (int i : idx.indices)
        cluster->push_back((*no_plane)[i]); // cluster point cloud

      // Compute OBB
      Eigen::Vector3f obb_pos;
      Eigen::Matrix3f obb_rot;
      Eigen::Vector3f obb_min, obb_max;
      if (!computeOBB(cluster, obb_pos, obb_rot, obb_min, obb_max)) // if OBB computation fails (e.g. too few points), skip this cluster
        continue;                                                   // does it look like a pallet? if NO -> skip
      // OBB dimensions sorted
      Eigen::Vector3f dims = (obb_max - obb_min).cwiseAbs(); // get dimensions of OBB (length, width, height) as absolute difference between max and min corners
      std::vector<float> d = {dims.x(), dims.y(), dims.z()}; // sort dimensions to be agnostic to cluster orientation (pallet could be rotated in any way)
      std::sort(d.begin(), d.end());                         // small -> large

      // Pallet dims sorted
      std::vector<double> pd = {pallet_H_, pallet_W_, pallet_L_};
      std::sort(pd.begin(), pd.end());

      // Gate: check size within tolerances
      if (std::abs(d[2] - pd[2]) > tol_L_)
        continue; // largest ~ length
      if (std::abs(d[1] - pd[1]) > tol_W_)
        continue; // middle ~ width
      if (std::abs(d[0] - pd[0]) > tol_H_)
        continue; // smallest ~ height

      // 6a) Deter mine OBB local axes by extent: thin=height, long=length, mid=width.
      int thin_axis = 0;
      int mid_axis = 1;
      int long_axis = 2;
      axisOrder(dims, thin_axis, mid_axis, long_axis);

      // 6b) Transform cluster points into OBB-local coordinates.
      // Local coordinates make all face/histogram checks view-angle invariant.
      std::vector<Eigen::Vector3f> local_pts;
      local_pts.reserve(cluster->size());
      for (const auto &pt : *cluster)
      {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        local_pts.push_back(obb_rot.transpose() * (p - obb_pos));
      }

      // 6c) Top-face occupancy score (soft cue) + optional hard gate.
      double top_fill_ratio = 0.0;
      const double top_score = scoreTopFace(local_pts, obb_min, obb_max, long_axis, mid_axis, top_fill_ratio);
      if (enable_top_fill_gate_ && (top_fill_ratio < min_fill_ratio_ || top_fill_ratio > max_fill_ratio_))
        continue;

      // 6d) Front/side structural score (main discriminative cue).
      double front_fill_ratio = 0.0;
      int front_void_cols = 0;
      const double front_score = scoreFrontStructure(local_pts, obb_min, obb_max, long_axis, mid_axis, thin_axis,
                                                     front_fill_ratio, front_void_cols);
      if (front_score < min_front_score_)
        continue;

      // 6e) Height-profile bimodality + dimension closeness.
      const double height_score = scoreHeightBimodality(local_pts, obb_min, obb_max, thin_axis);
      const double dim_score = scoreDimensions(d, pd);

      // 6f) Weighted fusion into a single cluster confidence score.
      const double total_score =
          w_dim_ * dim_score + w_top_ * top_score + w_front_ * front_score + w_height_ * height_score;
      if (total_score < min_total_score_)
        continue;

      ROS_INFO_STREAM_THROTTLE(1.0, "Cluster " << cluster_id
                                               << ": total=" << total_score
                                               << " dim=" << dim_score
                                               << " top=" << top_score << " (fill=" << top_fill_ratio << ")"
                                               << " front=" << front_score << " (fill=" << front_fill_ratio << ", void_cols=" << front_void_cols << ")"
                                               << " height=" << height_score
                                               << " dims=(" << d[0] << ", " << d[1] << ", " << d[2] << ")"
                                               << " pts=" << cluster->size()
                                               << " centroid=(" << obb_pos.x() << ", " << obb_pos.y() << ", " << obb_pos.z() << ")");

      if (total_score > best_total_score) // keep highest-confidence cluster in this frame
      {
        ROS_INFO_STREAM_THROTTLE(1.0, "Cluster " << cluster_id << " is new best (prev best_total=" << best_total_score << ")");
        best_total_score = total_score;
        best_dim_score = dim_score;
        best_top_score = top_score;
        best_front_score = front_score;
        best_height_score = height_score;
        best_top_fill_ratio = top_fill_ratio;
        best_front_fill_ratio = front_fill_ratio;
        best_front_void_cols = front_void_cols;
        candidate_found = true;

        // Pose from OBB
        best_pose.header = msg->header;
        best_pose.header.frame_id = frame;
        best_pose.pose.position.x = obb_pos.x();
        best_pose.pose.position.y = obb_pos.y();
        best_pose.pose.position.z = obb_pos.z();

        Eigen::Quaternionf q(obb_rot);
        best_pose.pose.orientation.x = q.x();
        best_pose.pose.orientation.y = q.y();
        best_pose.pose.orientation.z = q.z();
        best_pose.pose.orientation.w = q.w();

        // Marker (OBB)
        best_marker = makeOBBMarker(msg->header, frame, obb_pos, obb_rot, dims);
      }
      cluster_id++;
    }

    // 7) Temporal tracking — gate, smooth, and apply confidence hysteresis.
    // 7a) If we already track an object, reject implausible jumps and smooth accepted pose.
    if (candidate_found && has_track_)
    {
      // Gate: reject if candidate jumped too far from tracked pose (likely a different object)
      double dx = best_pose.pose.position.x - track_pose_.pose.position.x;
      double dy = best_pose.pose.position.y - track_pose_.pose.position.y;
      double dz = best_pose.pose.position.z - track_pose_.pose.position.z;
      double jump = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (jump > track_gate_dist_)
      {
        ROS_WARN_STREAM_THROTTLE(1.0, "Track gate rejected candidate: jump=" << jump << "m");
        candidate_found = false; // reject outlier jump, fall through to coast logic
      }
      else
      {
        // Exponential smoothing: blend new detection with previous tracked pose
        best_pose.pose.position.x = track_alpha_ * best_pose.pose.position.x + (1.0 - track_alpha_) * track_pose_.pose.position.x;
        best_pose.pose.position.y = track_alpha_ * best_pose.pose.position.y + (1.0 - track_alpha_) * track_pose_.pose.position.y;
        best_pose.pose.position.z = track_alpha_ * best_pose.pose.position.z + (1.0 - track_alpha_) * track_pose_.pose.position.z;
      }
    }

    // 7b) Update confidence state (EMA): rise on good detections, decay when missing.
    if (candidate_found)
    {
      track_hit_streak_++;
      track_confidence_ = (1.0 - track_conf_alpha_) * track_confidence_ + track_conf_alpha_ * best_total_score;
    }
    else
    {
      track_hit_streak_ = 0;
      track_confidence_ = (1.0 - track_conf_alpha_) * track_confidence_;
    }

    // 7c) Hysteresis rule:
    // - New track needs higher confidence + consecutive hits.
    // - Existing track can survive at a lower keep threshold.
    const bool confidence_ok = has_track_
                                   ? (track_confidence_ >= track_keep_conf_)
                                   : (track_confidence_ >= track_acquire_conf_ && track_hit_streak_ >= track_acquire_hits_);

    // 7d) Publish best pose/marker, coast on brief misses, or declare track lost.
    if (candidate_found && confidence_ok)
    {
      has_track_ = true;
      track_miss_count_ = 0;
      track_pose_ = best_pose;
      track_marker_ = best_marker;

      // Extract yaw from quaternion for readable logging
      double qx = best_pose.pose.orientation.x;
      double qy = best_pose.pose.orientation.y;
      double qz = best_pose.pose.orientation.z;
      double qw = best_pose.pose.orientation.w;
      double yaw_rad = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
      double yaw_deg = yaw_rad * 180.0 / M_PI;

      ROS_INFO_STREAM_THROTTLE(1.0, "PALLET DETECTED: total=" << best_total_score
                                                              << " [dim=" << best_dim_score
                                                              << ", top=" << best_top_score << " fill=" << best_top_fill_ratio
                                                              << ", front=" << best_front_score << " fill=" << best_front_fill_ratio
                                                              << " void_cols=" << best_front_void_cols
                                                              << ", height=" << best_height_score << "]"
                                                              << " conf=" << track_confidence_
                                                              << " pos=(" << best_pose.pose.position.x
                                                              << ", " << best_pose.pose.position.y
                                                              << ", " << best_pose.pose.position.z << ")"
                                                              << " yaw=" << yaw_deg << " deg");
      pose_pub_.publish(best_pose);
      marker_pub_.publish(best_marker);
    }
    else if (has_track_ && track_miss_count_ < track_max_miss_)
    {
      // Coast: keep publishing last known pose for a few frames during brief occlusions
      track_miss_count_++;
      ROS_WARN_STREAM_THROTTLE(1.0, "Coasting on track (" << track_miss_count_ << "/" << track_max_miss_
                                                          << "), conf=" << track_confidence_);
      pose_pub_.publish(track_pose_);
      marker_pub_.publish(track_marker_);
    }
    else
    {
      // Lost track entirely
      if (has_track_)
        ROS_WARN_STREAM("Track lost after " << track_max_miss_ << " missed frames");
      has_track_ = false;
      track_miss_count_ = 0;
      ROS_WARN_STREAM_THROTTLE(2.0, "No pallet published. clusters=" << cluster_indices.size()
                                                                     << " candidate_found=" << (candidate_found ? 1 : 0)
                                                                     << " conf=" << track_confidence_);
    }
  }

  // Helper functions for each processing step:
  // job of this function: take input cloud, apply crop box filter to limit to region of interest (ROI) where pallets are expected, return cropped cloud
  CloudT::Ptr cropBoxROI(const CloudT::Ptr &in) // crop box filter to limit point cloud to region of interest (ROI) where pallets are expected
  {
    // Axis convention (optical frame):
    //   x = left/right (side)
    //   y = up/down (vertical)
    //   z = forward (depth)
    CloudT::Ptr out(new CloudT);
    pcl::CropBox<PointT> crop;
    crop.setInputCloud(in);
    crop.setMin(Eigen::Vector4f(x_min_, y_min_, z_min_, 1.0f));
    crop.setMax(Eigen::Vector4f(x_max_, y_max_, z_max_, 1.0f));
    crop.filter(*out);
    return out;
  }

  // job of this function: take input cloud, apply RANSAC plane segmentation to find dominant plane (floor/table), remove inliers of that plane from cloud, return cloud without dominant plane
  void removeDominantPlane(const CloudT::Ptr &in, CloudT::Ptr &out)
  {
    pcl::SACSegmentation<PointT> seg;             // RANSAC plane segmentation to find dominant plane (floor/table)
    seg.setOptimizeCoefficients(true);            // optional: refine plane coefficients after segmentation
    seg.setModelType(pcl::SACMODEL_PLANE);        // looking for plane model
    seg.setMethodType(pcl::SAC_RANSAC);           // RANSAC method
    seg.setMaxIterations(plane_max_iters_);       // max RANSAC iterations
    seg.setDistanceThreshold(plane_dist_thresh_); // inlier distance threshold
    seg.setInputCloud(in);                        // segment largest plane in the cloud

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coeff);

    // If no plane found, return original
    if (inliers->indices.empty())
    {
      out = in;
      return;
    }

    // inliers refer to points that belong to the plane. We want to remove those to get the cloud without the dominant plane (floor/table)
    pcl::ExtractIndices<PointT> ex; // extract inliers to remove dominant plane points from cloud
    ex.setInputCloud(in);           // set input cloud
    ex.setIndices(inliers);         // set inliers to remove
    ex.setNegative(true);           // setNegative=true means KEEPING the points that are NOT in the plane
    ex.filter(*out);
  }

  // job of this function: take input cloud (with dominant plane removed), apply Euclidean clustering to find clusters of points that could be pallets, return vector of clusters (each cluster is vector of point indices)
  void euclideanClusters(const CloudT::Ptr &in, std::vector<pcl::PointIndices> &clusters) // Euclidean clus to find clusters of points (potential pallets) in the non-dominant-plane cloud
  {
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>); // KdTree for efficient neighbor search during clustering
    tree->setInputCloud(in);

    pcl::EuclideanClusterExtraction<PointT> ec; // main PCL function for Euclidean clustering
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(cluster_min_size_);
    ec.setMaxClusterSize(cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in);
    ec.extract(clusters);
  }

  // job of this function: clamp any value to [0,1] for normalized scoring safety.
  static double clamp01(double x)
  {
    return std::max(0.0, std::min(1.0, x));
  }

  // job of this function: decide which OBB axis is thin/mid/long based on extents.
  // This lets later checks use "height/width/length" semantics regardless of object orientation.
  void axisOrder(const Eigen::Vector3f &dims, int &thin_axis, int &mid_axis, int &long_axis) const
  {
    thin_axis = 0;
    if (dims.y() < dims[thin_axis])
      thin_axis = 1;
    if (dims.z() < dims[thin_axis])
      thin_axis = 2;

    long_axis = 0;
    if (dims.y() > dims[long_axis])
      long_axis = 1;
    if (dims.z() > dims[long_axis])
      long_axis = 2;

    mid_axis = 3 - thin_axis - long_axis;
  }

  // job of this function: project points onto OBB top face and compute fill-ratio-based score.
  // Returns:
  // - fill_ratio: occupied_cells / all_cells
  // - score in [0,1]: highest near configured pallet-like fill range center.
  double scoreTopFace(const std::vector<Eigen::Vector3f> &local_pts,
                      const Eigen::Vector3f &obb_min,
                      const Eigen::Vector3f &obb_max,
                      int axis_a,
                      int axis_b,
                      double &fill_ratio) const
  {
    const double extent_a = std::max(1e-6, static_cast<double>(obb_max[axis_a] - obb_min[axis_a]));
    const double extent_b = std::max(1e-6, static_cast<double>(obb_max[axis_b] - obb_min[axis_b]));
    const int grid_a = std::max(1, static_cast<int>(std::ceil(extent_a / occupancy_cell_size_)));
    const int grid_b = std::max(1, static_cast<int>(std::ceil(extent_b / occupancy_cell_size_)));
    std::vector<unsigned char> grid(grid_a * grid_b, 0);

    int occupied = 0;
    for (const auto &p : local_pts)
    {
      const int ia = static_cast<int>((static_cast<double>(p[axis_a]) - obb_min[axis_a]) / occupancy_cell_size_);
      const int ib = static_cast<int>((static_cast<double>(p[axis_b]) - obb_min[axis_b]) / occupancy_cell_size_);
      if (ia < 0 || ia >= grid_a || ib < 0 || ib >= grid_b)
        continue;
      const int k = ia * grid_b + ib;
      if (!grid[k])
      {
        grid[k] = 1;
        occupied++;
      }
    }

    fill_ratio = static_cast<double>(occupied) / static_cast<double>(grid_a * grid_b);
    const double ctr = 0.5 * (min_fill_ratio_ + max_fill_ratio_);
    const double half = std::max(1e-6, 0.5 * (max_fill_ratio_ - min_fill_ratio_));
    return clamp01(1.0 - std::abs(fill_ratio - ctr) / half);
  }

  // job of this function: score fork-pocket-like structure on vertical faces.
  // It evaluates both long×height and mid×height faces, then keeps the better one.
  // A good pallet face should have:
  // - moderate overall fill (not empty, not fully solid)
  // - lower-band void columns (pocket openings)
  // - sufficient upper-band occupancy (deck/stringer structure)
  double scoreFrontStructure(const std::vector<Eigen::Vector3f> &local_pts,
                             const Eigen::Vector3f &obb_min,
                             const Eigen::Vector3f &obb_max,
                             int long_axis,
                             int mid_axis,
                             int thin_axis,
                             double &best_fill_ratio,
                             int &best_void_cols) const
  {
    auto eval_face = [&](int face_axis, double &fill_ratio, int &void_cols) -> double
    {
      const double extent_h = std::max(1e-6, static_cast<double>(obb_max[face_axis] - obb_min[face_axis]));
      const double extent_t = std::max(1e-6, static_cast<double>(obb_max[thin_axis] - obb_min[thin_axis]));
      const int grid_h = std::max(1, static_cast<int>(std::ceil(extent_h / front_cell_size_)));
      const int grid_t = std::max(1, static_cast<int>(std::ceil(extent_t / front_cell_size_)));
      std::vector<unsigned char> grid(grid_h * grid_t, 0);

      int occupied = 0;
      for (const auto &p : local_pts)
      {
        const int ih = static_cast<int>((static_cast<double>(p[face_axis]) - obb_min[face_axis]) / front_cell_size_);
        const int it = static_cast<int>((static_cast<double>(p[thin_axis]) - obb_min[thin_axis]) / front_cell_size_);
        if (ih < 0 || ih >= grid_h || it < 0 || it >= grid_t)
          continue;
        const int k = ih * grid_t + it;
        if (!grid[k])
        {
          grid[k] = 1;
          occupied++;
        }
      }

      fill_ratio = static_cast<double>(occupied) / static_cast<double>(grid_h * grid_t);

      int lower_rows = std::max(1, static_cast<int>(std::round(front_lower_height_ratio_ * grid_t)));
      lower_rows = std::min(lower_rows, grid_t);

      int lower_occ_total = 0;
      void_cols = 0;
      for (int ih = 0; ih < grid_h; ++ih)
      {
        int col_lower = 0;
        for (int it = 0; it < lower_rows; ++it)
        {
          if (grid[ih * grid_t + it])
            col_lower++;
        }
        lower_occ_total += col_lower;
        const double col_fill = static_cast<double>(col_lower) / static_cast<double>(lower_rows);
        if (col_fill <= front_void_column_fill_max_)
          void_cols++;
      }

      int upper_occ_total = 0;
      int upper_cells = 0;
      for (int ih = 0; ih < grid_h; ++ih)
      {
        for (int it = lower_rows; it < grid_t; ++it)
        {
          if (grid[ih * grid_t + it])
            upper_occ_total++;
          upper_cells++;
        }
      }
      const double lower_fill = static_cast<double>(lower_occ_total) / static_cast<double>(grid_h * lower_rows);
      const double upper_fill = (upper_cells > 0) ? static_cast<double>(upper_occ_total) / static_cast<double>(upper_cells)
                                                  : fill_ratio;

      const double fill_ctr = 0.5 * (front_fill_min_ + front_fill_max_);
      const double fill_half = std::max(1e-6, 0.5 * (front_fill_max_ - front_fill_min_));
      const double fill_score = clamp01(1.0 - std::abs(fill_ratio - fill_ctr) / fill_half);
      const double void_score = (front_min_void_columns_ > 0)
                                    ? clamp01(static_cast<double>(void_cols) / static_cast<double>(front_min_void_columns_))
                                    : 1.0;
      const double upper_score =
          clamp01((upper_fill - front_upper_fill_min_) / std::max(1e-6, 1.0 - front_upper_fill_min_));
      const double contrast_score = clamp01((upper_fill - lower_fill + 0.05) / 0.45);

      return 0.35 * fill_score + 0.40 * void_score + 0.15 * upper_score + 0.10 * contrast_score;
    };

    double fill_long = 0.0;
    double fill_mid = 0.0;
    int void_long = 0;
    int void_mid = 0;
    const double score_long = eval_face(long_axis, fill_long, void_long);
    const double score_mid = eval_face(mid_axis, fill_mid, void_mid);

    if (score_long >= score_mid)
    {
      best_fill_ratio = fill_long;
      best_void_cols = void_long;
      return score_long;
    }

    best_fill_ratio = fill_mid;
    best_void_cols = void_mid;
    return score_mid;
  }

  // job of this function: score bimodality of occupancy along OBB thin axis (height direction).
  // Pallets often show two main strata; solid boxes trend toward flatter/single-peak profiles.
  double scoreHeightBimodality(const std::vector<Eigen::Vector3f> &local_pts,
                               const Eigen::Vector3f &obb_min,
                               const Eigen::Vector3f &obb_max,
                               int thin_axis) const
  {
    const int bins = std::max(6, height_bins_);
    const double thin_extent = std::max(1e-6, static_cast<double>(obb_max[thin_axis] - obb_min[thin_axis]));
    std::vector<double> hist(bins, 0.0);

    for (const auto &p : local_pts)
    {
      int b = static_cast<int>((static_cast<double>(p[thin_axis]) - obb_min[thin_axis]) / thin_extent * bins);
      b = std::max(0, std::min(bins - 1, b));
      hist[b] += 1.0;
    }

    // Light smoothing to reduce sensor noise.
    std::vector<double> smooth(bins, 0.0);
    for (int i = 0; i < bins; ++i)
    {
      double sum = hist[i];
      int cnt = 1;
      if (i > 0)
      {
        sum += hist[i - 1];
        cnt++;
      }
      if (i + 1 < bins)
      {
        sum += hist[i + 1];
        cnt++;
      }
      smooth[i] = sum / static_cast<double>(cnt);
    }

    const int split = bins / 2;
    int low_idx = 0;
    int high_idx = split;
    for (int i = 1; i < split; ++i)
      if (smooth[i] > smooth[low_idx])
        low_idx = i;
    for (int i = split + 1; i < bins; ++i)
      if (smooth[i] > smooth[high_idx])
        high_idx = i;

    const double peak_low = smooth[low_idx];
    const double peak_high = smooth[high_idx];
    const double peak_max = std::max(peak_low, peak_high);
    if (peak_max <= 1e-6)
      return 0.0;

    int a = low_idx;
    int b = high_idx;
    if (a > b)
      std::swap(a, b);
    double valley = peak_max;
    for (int i = a; i <= b; ++i)
      valley = std::min(valley, smooth[i]);

    const double valley_ratio = valley / peak_max;
    const double valley_score =
        clamp01((height_valley_ratio_max_ - valley_ratio) / std::max(1e-6, height_valley_ratio_max_));
    const double balance = std::min(peak_low, peak_high) / peak_max;
    const double balance_score = clamp01(balance / 0.6);
    return clamp01(valley_score * (0.5 + 0.5 * balance_score));
  }

  // job of this function: convert dimension errors into a normalized [0,1] score.
  // 1.0 means close to expected pallet dimensions; 0.0 means near/at tolerance limits.
  double scoreDimensions(const std::vector<float> &d_sorted, const std::vector<double> &pd_sorted) const
  {
    const double e_h = std::abs(d_sorted[0] - pd_sorted[0]) / std::max(1e-6, tol_H_);
    const double e_w = std::abs(d_sorted[1] - pd_sorted[1]) / std::max(1e-6, tol_W_);
    const double e_l = std::abs(d_sorted[2] - pd_sorted[2]) / std::max(1e-6, tol_L_);
    const double norm_err = (clamp01(e_h) + clamp01(e_w) + clamp01(e_l)) / 3.0;
    return clamp01(1.0 - norm_err);
  }

  // COMPUTATION OF OBB (ORIENTED BOUNDING BOX) USING PCA + MINIMUM-AREA BOUNDING RECTANGLE
  // PCA finds the thin axis (pallet height) reliably regardless of viewing angle.
  // For the in-plane orientation (length/width), we project points onto the top face,
  // compute the 2D convex hull, and find the minimum-area bounding rectangle.
  // This is immune to point density bias from perspective viewing angles.
  bool computeOBB(const CloudT::Ptr &cluster,
                  Eigen::Vector3f &obb_pos,
                  Eigen::Matrix3f &obb_rot,
                  Eigen::Vector3f &obb_min,
                  Eigen::Vector3f &obb_max)
  {
    if (cluster->size() < 50)
      return false;

    // --- Step 1: PCA to find the thin axis (smallest eigenvalue = pallet height) ---
    Eigen::Vector4f centroid4;
    pcl::compute3DCentroid(*cluster, centroid4);
    Eigen::Vector3f centroid = centroid4.head<3>();

    Eigen::Matrix3f cov;
    pcl::computeCovarianceMatrixNormalized(*cluster, centroid4, cov);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    Eigen::Matrix3f evecs = solver.eigenvectors(); // columns sorted ascending by eigenvalue

    Eigen::Vector3f thin_axis = evecs.col(0); // smallest variance = height direction
    Eigen::Vector3f u_axis = evecs.col(1);    // in-plane axis (PCA, will be refined)
    Eigen::Vector3f v_axis = evecs.col(2);    // in-plane axis (PCA, will be refined)

    // --- Step 2: Project points onto the top-face plane (u, v) and record thin-axis coords ---
    std::vector<Eigen::Vector2f> pts2d(cluster->size());
    std::vector<float> thin_vals(cluster->size());

    for (size_t i = 0; i < cluster->size(); ++i)
    {
      Eigen::Vector3f p((*cluster)[i].x, (*cluster)[i].y, (*cluster)[i].z);
      Eigen::Vector3f d = p - centroid;
      pts2d[i] = Eigen::Vector2f(d.dot(u_axis), d.dot(v_axis));
      thin_vals[i] = d.dot(thin_axis);
    }

    // --- Step 3: 2D Convex Hull (Andrew's monotone chain) ---
    std::vector<size_t> order(pts2d.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
              {
      if (pts2d[a].x() != pts2d[b].x()) return pts2d[a].x() < pts2d[b].x();
      return pts2d[a].y() < pts2d[b].y(); });

    auto cross2d = [](const Eigen::Vector2f &O, const Eigen::Vector2f &A, const Eigen::Vector2f &B) -> float
    {
      return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
    };

    std::vector<Eigen::Vector2f> hull;
    // Lower hull
    for (size_t i : order)
    {
      while (hull.size() >= 2 && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts2d[i]) <= 0.0f)
        hull.pop_back();
      hull.push_back(pts2d[i]);
    }
    // Upper hull
    size_t lower_sz = hull.size() + 1;
    for (auto it = order.rbegin(); it != order.rend(); ++it)
    {
      while (hull.size() >= lower_sz && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts2d[*it]) <= 0.0f)
        hull.pop_back();
      hull.push_back(pts2d[*it]);
    }
    hull.pop_back(); // remove duplicate of first point

    if (hull.size() < 3)
      return false;

    // --- Step 4: Minimum-area bounding rectangle via rotating calipers ---
    float best_area = FLT_MAX;
    float best_angle = 0.0f;
    Eigen::Vector2f best_rmin, best_rmax;

    for (size_t i = 0; i < hull.size(); ++i)
    {
      size_t j = (i + 1) % hull.size();
      float angle = std::atan2(hull[j].y() - hull[i].y(), hull[j].x() - hull[i].x());
      float ca = std::cos(angle), sa = std::sin(angle);

      Eigen::Vector2f mn(FLT_MAX, FLT_MAX);
      Eigen::Vector2f mx(-FLT_MAX, -FLT_MAX);

      for (const auto &p : hull)
      {
        // Rotate by -angle to align this edge with the X axis
        float rx = ca * p.x() + sa * p.y();
        float ry = -sa * p.x() + ca * p.y();
        mn.x() = std::min(mn.x(), rx);
        mn.y() = std::min(mn.y(), ry);
        mx.x() = std::max(mx.x(), rx);
        mx.y() = std::max(mx.y(), ry);
      }

      float area = (mx.x() - mn.x()) * (mx.y() - mn.y());
      if (area < best_area)
      {
        best_area = area;
        best_angle = angle;
        best_rmin = mn;
        best_rmax = mx;
      }
    }

    // --- Step 5: Build OBB from the minimum-area rectangle ---
    float ca = std::cos(best_angle), sa = std::sin(best_angle);

    // Refined in-plane axes aligned with the minimum-area rectangle (world frame)
    Eigen::Vector3f new_u = ca * u_axis + sa * v_axis;
    Eigen::Vector3f new_v = -sa * u_axis + ca * v_axis;

    // Ensure right-handed coordinate system
    if (thin_axis.dot(new_u.cross(new_v)) < 0)
      thin_axis = -thin_axis;

    // Rectangle center in rotated 2D frame → transform back to u,v frame
    Eigen::Vector2f rc = 0.5f * (best_rmin + best_rmax);
    float cx_uv = ca * rc.x() - sa * rc.y(); // inverse of the -angle rotation
    float cy_uv = sa * rc.x() + ca * rc.y();

    // Thin-axis extents
    float thin_min = *std::min_element(thin_vals.begin(), thin_vals.end());
    float thin_max = *std::max_element(thin_vals.begin(), thin_vals.end());
    float thin_ctr = 0.5f * (thin_min + thin_max);

    // OBB center in world coordinates
    obb_pos = centroid + cx_uv * u_axis + cy_uv * v_axis + thin_ctr * thin_axis;

    // OBB rotation: columns are the three OBB axes
    obb_rot.col(0) = new_u;
    obb_rot.col(1) = new_v;
    obb_rot.col(2) = thin_axis;

    // OBB half-extents in local frame (centered at origin)
    float half_u = 0.5f * (best_rmax.x() - best_rmin.x());
    float half_v = 0.5f * (best_rmax.y() - best_rmin.y());
    float half_t = 0.5f * (thin_max - thin_min);

    obb_min = Eigen::Vector3f(-half_u, -half_v, -half_t);
    obb_max = Eigen::Vector3f(half_u, half_v, half_t);

    return true;
  }

  // job of this function: take OBB parameters, create a visualization marker (green box) that can be published and visualized in RViz to show the detected pallet's position and orientation
  visualization_msgs::Marker makeOBBMarker(const std_msgs::Header &hdr,
                                           const std::string &frame,
                                           const Eigen::Vector3f &pos,
                                           const Eigen::Matrix3f &rot,
                                           const Eigen::Vector3f &dims)
  {
    visualization_msgs::Marker m;
    m.header = hdr;
    m.header.frame_id = frame;
    m.ns = "pallet_detector";
    m.id = 0;
    m.type = visualization_msgs::Marker::CUBE;
    m.action = visualization_msgs::Marker::ADD;

    m.pose.position.x = pos.x();
    m.pose.position.y = pos.y();
    m.pose.position.z = pos.z();

    Eigen::Quaternionf q(rot);
    m.pose.orientation.x = q.x();
    m.pose.orientation.y = q.y();
    m.pose.orientation.z = q.z();
    m.pose.orientation.w = q.w();

    m.scale.x = dims.x();
    m.scale.y = dims.y();
    m.scale.z = dims.z();

    m.color.r = 0.1f;
    m.color.g = 0.9f;
    m.color.b = 0.1f;
    m.color.a = 0.4f;
    m.lifetime = ros::Duration(0.1);
    return m;
  }

  // Member variables for ROS interfaces, parameters, and internal state
private:
  ros::NodeHandle nh_, pnh_;                                      // ROS node handles for public and private namespaces
  ros::Subscriber sub_;                                           // the "ears" -> listens to incoming point clouds
  ros::Publisher pose_pub_, marker_pub_, roi_pub_, no_plane_pub_; // the "mouth" -> publishes in rviz detected pallet pose and visualization marker, also intermediate clouds for debugging

  std::string cloud_topic_; // ROS topic to subscribe for input point cloud
  std::string output_frame_;

  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_; // ROI crop boundaries
  double voxel_leaf_;                                    // how aggressively to downsample
  bool use_sor_;                                         // whether to apply statistical outlier removal
  int sor_mean_k_;                                       // number of neighbors to analyze for each point in SOR
  double sor_stddev_;                                    //

  double plane_dist_thresh_; // how thick the floor can be (distance threshold for plane in RANSAC)
  int plane_max_iters_;      //

  double cluster_tolerance_; // how  close points must be to group together in clustering
  int cluster_min_size_;
  int cluster_max_size_;

  double pallet_L_, pallet_W_, pallet_H_; // expected dimensions of a pallet (length, width, height)
  double tol_L_, tol_W_, tol_H_;          // tolerances for how much detected cluster dimensions can deviate from expected pallet dimensions to still be considered a match
  double occupancy_cell_size_;            // top-face grid cell size
  double min_fill_ratio_;                 // nominal top-face fill lower bound
  double max_fill_ratio_;                 // nominal top-face fill upper bound
  bool enable_top_fill_gate_;             // optional hard gate on top fill

  // Structure scoring parameters
  double front_cell_size_; // front/side face grid cell size
  double front_fill_min_;
  double front_fill_max_;
  double front_lower_height_ratio_;   // lower-height band where fork pockets are expected
  double front_void_column_fill_max_; // a "void column" has lower-band fill below this
  int front_min_void_columns_;        // minimum number of void columns for pocket-like face
  double front_upper_fill_min_;       // top part of front face should have some occupancy

  // Height bimodality parameters
  int height_bins_;
  double height_valley_ratio_max_;

  // Weighted fusion
  double w_dim_;
  double w_top_;
  double w_front_;
  double w_height_;
  double min_front_score_;
  double min_total_score_;

  // Temporal tracking state
  double track_gate_dist_;                  // max allowed position jump (m) to accept as same pallet
  int track_max_miss_;                      // how many frames to coast on last known pose before dropping track
  double track_alpha_;                      // exponential smoothing factor (0=full old, 1=full new)
  double track_conf_alpha_;                 // confidence smoothing
  double track_acquire_conf_;               // confidence needed to start a new track
  double track_keep_conf_;                  // confidence needed to keep current track
  int track_acquire_hits_;                  // consecutive candidate hits required to acquire track
  bool has_track_ = false;                  // whether we currently have an active track
  int track_miss_count_ = 0;                // consecutive frames with no detection
  int track_hit_streak_ = 0;                // consecutive candidate frames (for acquisition hysteresis)
  double track_confidence_ = 0.0;           // fused confidence [0, 1]
  geometry_msgs::PoseStamped track_pose_;   // last known smoothed pose
  visualization_msgs::Marker track_marker_; // last known marker (for coasting)
};

// Main function to initialize ROS node and start processing
int main(int argc, char **argv)
{
  ros::init(argc, argv, "pallet_detector");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  PalletDetector node(nh, pnh);
  ros::spin();
  return 0;
}
