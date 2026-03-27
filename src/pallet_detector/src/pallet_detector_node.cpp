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

    // Occupancy ratio check: project cluster onto the OBB top face (two largest dims)
    // and measure what fraction of grid cells are occupied.
    // A pallet with gaps between deck boards has a lower fill ratio (~0.4–0.75)
    // than a solid box of the same size (~0.85–1.0).
    pnh_.param("occupancy_cell_size", occupancy_cell_size_, 0.02); // grid cell size in meters
    pnh_.param("min_fill_ratio", min_fill_ratio_, 0.3);            // below this = too sparse
    pnh_.param("max_fill_ratio", max_fill_ratio_, 0.8);            // above this = too solid (no gaps)

    // Temporal tracking — smooth pose output and coast through brief occlusions
    pnh_.param("track_gate_dist", track_gate_dist_, 0.3); // max allowed jump (m) between frames to accept as same pallet
    pnh_.param("track_max_miss", track_max_miss_, 5);     // frames to keep publishing last pose when detection drops
    pnh_.param("track_alpha", track_alpha_, 0.3);         // exponential smoothing factor (0=full old, 1=full new)

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

    // 6) Score clusters by geometry, pick best
    // Initialize best result tracking
    bool found = false;
    geometry_msgs::PoseStamped best_pose;
    visualization_msgs::Marker best_marker;
    double best_score = 1e9;

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

      // Occupancy ratio check: project points onto OBB top face, measure fill ratio.
      // A pallet has gaps between deck boards → lower fill ratio than a solid box.
      {
        // Find the thin axis (smallest dimension = pallet height)
        int thin_axis = 0;
        if (dims.y() < dims.x() && dims.y() < dims.z()) thin_axis = 1;
        else if (dims.z() < dims.x() && dims.z() < dims.y()) thin_axis = 2;

        // The other two axes form the top face
        int axis_a = (thin_axis == 0) ? 1 : 0;
        int axis_b = (thin_axis == 2) ? 1 : 2;

        double extent_a = std::abs(obb_max[axis_a] - obb_min[axis_a]);
        double extent_b = std::abs(obb_max[axis_b] - obb_min[axis_b]);

        int grid_a = std::max(1, static_cast<int>(std::ceil(extent_a / occupancy_cell_size_)));
        int grid_b = std::max(1, static_cast<int>(std::ceil(extent_b / occupancy_cell_size_)));

        std::vector<bool> grid(grid_a * grid_b, false);

        for (const auto &pt : *cluster)
        {
          Eigen::Vector3f p(pt.x, pt.y, pt.z);
          Eigen::Vector3f local = obb_rot.transpose() * (p - obb_pos);

          int ia = static_cast<int>((static_cast<double>(local[axis_a]) - obb_min[axis_a]) / occupancy_cell_size_);
          int ib = static_cast<int>((static_cast<double>(local[axis_b]) - obb_min[axis_b]) / occupancy_cell_size_);

          if (ia >= 0 && ia < grid_a && ib >= 0 && ib < grid_b)
            grid[ia * grid_b + ib] = true;
        }

        int occupied = std::count(grid.begin(), grid.end(), true);
        double fill_ratio = static_cast<double>(occupied) / static_cast<double>(grid_a * grid_b);

        ROS_INFO_STREAM("Cluster " << cluster_id << ": pts=" << cluster->size()
                                   << " dims=(" << d[0] << ", " << d[1] << ", " << d[2] << ")"
                                   << " fill_ratio=" << fill_ratio
                                   << " range=[" << min_fill_ratio_ << ", " << max_fill_ratio_ << "]"
                                   << " centroid=(" << obb_pos.x() << ", " << obb_pos.y() << ", " << obb_pos.z() << ")");

        if (fill_ratio < min_fill_ratio_ || fill_ratio > max_fill_ratio_)
          continue; // too sparse or too solid — not a pallet with deck board gaps
      }

      // Score: sum absolute errors
      double score = std::abs(d[2] - pd[2]) + std::abs(d[1] - pd[1]) + std::abs(d[0] - pd[0]); // if YES -> calculate a score
      ROS_INFO_STREAM("Cluster " << cluster_id << " PASSED all gates: score=" << score
                                 << " dims=(" << d[0] << ", " << d[1] << ", " << d[2] << ")"
                                 << " pts=" << cluster->size()
                                 << " centroid=(" << obb_pos.x() << ", " << obb_pos.y() << ", " << obb_pos.z() << ")");
      if (score < best_score) // is this score better than previous best_score?
      {
        ROS_INFO_STREAM("Cluster " << cluster_id << " is new best (prev best_score=" << best_score << ")");
        best_score = score; // if YES → save as new best
        found = true;       // flip found = true

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

    // 7) Temporal tracking — smooth pose and coast through brief occlusions
    if (found && has_track_)
    {
      // Gate: reject if candidate jumped too far from tracked pose (likely a different object)
      double dx = best_pose.pose.position.x - track_pose_.pose.position.x;
      double dy = best_pose.pose.position.y - track_pose_.pose.position.y;
      double dz = best_pose.pose.position.z - track_pose_.pose.position.z;
      double jump = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (jump > track_gate_dist_)
      {
        ROS_WARN_STREAM_THROTTLE(1.0, "Track gate rejected candidate: jump=" << jump << "m");
        found = false; // reject outlier jump, fall through to coast logic
      }
      else
      {
        // Exponential smoothing: blend new detection with previous tracked pose
        best_pose.pose.position.x = track_alpha_ * best_pose.pose.position.x + (1.0 - track_alpha_) * track_pose_.pose.position.x;
        best_pose.pose.position.y = track_alpha_ * best_pose.pose.position.y + (1.0 - track_alpha_) * track_pose_.pose.position.y;
        best_pose.pose.position.z = track_alpha_ * best_pose.pose.position.z + (1.0 - track_alpha_) * track_pose_.pose.position.z;
      }
    }

    // Publish best pose + marker, update track state
    if (found)
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

      ROS_INFO_STREAM_THROTTLE(1.0, "PALLET DETECTED: score=" << best_score
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
      ROS_WARN_STREAM_THROTTLE(1.0, "Pallet not found, coasting on track (" << track_miss_count_ << "/" << track_max_miss_ << ")");
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
      ROS_WARN_STREAM_THROTTLE(2.0, "No pallet found. clusters=" << cluster_indices.size()
                                                                 << " (check dims/tolerances or cluster_min_size)");
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
    Eigen::Vector3f u_axis    = evecs.col(1); // in-plane axis (PCA, will be refined)
    Eigen::Vector3f v_axis    = evecs.col(2); // in-plane axis (PCA, will be refined)

    // --- Step 2: Project points onto the top-face plane (u, v) and record thin-axis coords ---
    std::vector<Eigen::Vector2f> pts2d(cluster->size());
    std::vector<float> thin_vals(cluster->size());

    for (size_t i = 0; i < cluster->size(); ++i)
    {
      Eigen::Vector3f p((*cluster)[i].x, (*cluster)[i].y, (*cluster)[i].z);
      Eigen::Vector3f d = p - centroid;
      pts2d[i]    = Eigen::Vector2f(d.dot(u_axis), d.dot(v_axis));
      thin_vals[i] = d.dot(thin_axis);
    }

    // --- Step 3: 2D Convex Hull (Andrew's monotone chain) ---
    std::vector<size_t> order(pts2d.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      if (pts2d[a].x() != pts2d[b].x()) return pts2d[a].x() < pts2d[b].x();
      return pts2d[a].y() < pts2d[b].y();
    });

    auto cross2d = [](const Eigen::Vector2f &O, const Eigen::Vector2f &A, const Eigen::Vector2f &B) -> float {
      return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
    };

    std::vector<Eigen::Vector2f> hull;
    // Lower hull
    for (size_t i : order) {
      while (hull.size() >= 2 && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts2d[i]) <= 0.0f)
        hull.pop_back();
      hull.push_back(pts2d[i]);
    }
    // Upper hull
    size_t lower_sz = hull.size() + 1;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      while (hull.size() >= lower_sz && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts2d[*it]) <= 0.0f)
        hull.pop_back();
      hull.push_back(pts2d[*it]);
    }
    hull.pop_back(); // remove duplicate of first point

    if (hull.size() < 3)
      return false;

    // --- Step 4: Minimum-area bounding rectangle via rotating calipers ---
    float best_area  = FLT_MAX;
    float best_angle = 0.0f;
    Eigen::Vector2f best_rmin, best_rmax;

    for (size_t i = 0; i < hull.size(); ++i)
    {
      size_t j = (i + 1) % hull.size();
      float angle = std::atan2(hull[j].y() - hull[i].y(), hull[j].x() - hull[i].x());
      float ca = std::cos(angle), sa = std::sin(angle);

      Eigen::Vector2f mn( FLT_MAX,  FLT_MAX);
      Eigen::Vector2f mx(-FLT_MAX, -FLT_MAX);

      for (const auto &p : hull)
      {
        // Rotate by -angle to align this edge with the X axis
        float rx =  ca * p.x() + sa * p.y();
        float ry = -sa * p.x() + ca * p.y();
        mn.x() = std::min(mn.x(), rx);  mn.y() = std::min(mn.y(), ry);
        mx.x() = std::max(mx.x(), rx);  mx.y() = std::max(mx.y(), ry);
      }

      float area = (mx.x() - mn.x()) * (mx.y() - mn.y());
      if (area < best_area)
      {
        best_area  = area;
        best_angle = angle;
        best_rmin  = mn;
        best_rmax  = mx;
      }
    }

    // --- Step 5: Build OBB from the minimum-area rectangle ---
    float ca = std::cos(best_angle), sa = std::sin(best_angle);

    // Refined in-plane axes aligned with the minimum-area rectangle (world frame)
    Eigen::Vector3f new_u =  ca * u_axis + sa * v_axis;
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
    obb_max = Eigen::Vector3f( half_u,  half_v,  half_t);

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
  double occupancy_cell_size_;             // grid cell size for occupancy ratio check
  double min_fill_ratio_;                 // min fill ratio (below = too sparse)
  double max_fill_ratio_;                 // max fill ratio (above = too solid, no gaps)

  // Temporal tracking state
  double track_gate_dist_;                  // max allowed position jump (m) to accept as same pallet
  int track_max_miss_;                      // how many frames to coast on last known pose before dropping track
  double track_alpha_;                      // exponential smoothing factor (0=full old, 1=full new)
  bool has_track_ = false;                  // whether we currently have an active track
  int track_miss_count_ = 0;                // consecutive frames with no detection
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