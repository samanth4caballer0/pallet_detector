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

#include <pcl/features/moment_of_inertia_estimation.h>

#include <Eigen/Dense>
#include <algorithm>
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

    pnh_.param("max_surface_density", max_surface_density_, 800.0); // pts/m² — surface density; loose gate, sub-cluster check does the real filtering

    // Sub-cluster check: a real pallet has deck boards separated by gaps.
    // Re-clustering inside a candidate with a tight tolerance should yield multiple sub-clusters (boards).
    pnh_.param("subcluster_tolerance", subcluster_tolerance_, 0.03); // tighter than main clustering to split at gaps between boards
    pnh_.param("subcluster_min_size", subcluster_min_size_, 30);     // min points per sub-cluster (one board)
    pnh_.param("subcluster_min_count", subcluster_min_count_, 5);    // min number of sub-clusters expected (deck boards)

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

      // Surface density check: pts / surface_area (m²). A pallet has gaps in the top face
      // so its surface density is lower than a solid box of the same OBB dimensions.
      double surface_area = 2.0 * (static_cast<double>(d[0]) * d[1] + d[1] * d[2] + d[0] * d[2]);
      if (surface_area > 0.0)
      {
        double surf_density = static_cast<double>(cluster->size()) / surface_area;
        ROS_INFO_STREAM("Cluster: pts=" << cluster->size()
                                        << " dims=(" << d[0] << ", " << d[1] << ", " << d[2] << ")"
                                        << " surf_area=" << surface_area
                                        << " surf_density=" << surf_density
                                        << " threshold=" << max_surface_density_);
        if (surf_density > max_surface_density_)
          continue; // too dense surface — likely a solid object, not a pallet
      }

      // Sub-cluster check: re-cluster inside this candidate with a tighter tolerance.
      // A real pallet's deck boards + gaps should split into multiple sub-clusters.
      // A solid box stays as one sub-cluster.
      {
        pcl::search::KdTree<PointT>::Ptr sub_tree(new pcl::search::KdTree<PointT>);
        sub_tree->setInputCloud(cluster);
        pcl::EuclideanClusterExtraction<PointT> sub_ec;
        sub_ec.setClusterTolerance(subcluster_tolerance_);
        sub_ec.setMinClusterSize(subcluster_min_size_);
        sub_ec.setMaxClusterSize(static_cast<int>(cluster->size()));
        sub_ec.setSearchMethod(sub_tree);
        sub_ec.setInputCloud(cluster);
        std::vector<pcl::PointIndices> sub_clusters;
        sub_ec.extract(sub_clusters);

        ROS_INFO_STREAM("Sub-cluster check: " << sub_clusters.size()
                                              << " sub-clusters (need >=" << subcluster_min_count_ << ")"
                                              << " centroid=(" << obb_pos.x() << ", " << obb_pos.y() << ", " << obb_pos.z() << ")");
        if (static_cast<int>(sub_clusters.size()) < subcluster_min_count_)
          continue; // too few sub-clusters — solid object, not a pallet with deck boards
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

  // job of this function: take input cluster (point cloud), compute oriented bounding box (OBB) using PCL's moment of inertia estimation, return OBB parameters (position, rotation, min/max corners)
  bool computeOBB(const CloudT::Ptr &cluster,
                  Eigen::Vector3f &obb_pos,
                  Eigen::Matrix3f &obb_rot,
                  Eigen::Vector3f &obb_min,
                  Eigen::Vector3f &obb_max)
  {
    if (cluster->size() < 50) // too few points to compute reliable OBB
      return false;

    pcl::MomentOfInertiaEstimation<PointT> moi; // PCL class for computing oriented bounding box (OBB) using moment of inertia estimation
    moi.setInputCloud(cluster);                 // set input cloud for OBB computation (the cluster point cloud)
    moi.compute();
    // check if MoI uses PCA to compute OBB (if not, maybe subsititute it?)

    // first allocate variables to hold OBB outputs, then call getOBB() to compute OBB and fill those variables
    pcl::PointXYZ min_pt, max_pt, pos_pt;    // outputs of OBB computation: min_pt and max_pt are the corners of the bounding box, pos_pt is the center position of the box, rot is the rotation matrix representing the orientation of the box
    Eigen::Matrix3f rot;                     // convert rotation from PCL format to Eigen
    moi.getOBB(min_pt, max_pt, pos_pt, rot); // PCL function to compute OBB and return parameters

    obb_pos = pos_pt.getVector3fMap();
    obb_rot = rot;
    obb_min = min_pt.getVector3fMap();
    obb_max = max_pt.getVector3fMap();
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
  double max_surface_density_;            // max surface density (pts/m²) — reject clusters with denser surface than a pallet
  double subcluster_tolerance_;           // tight clustering tolerance to split deck boards at gaps
  int subcluster_min_size_;               // min points per sub-cluster (one board)
  int subcluster_min_count_;              // min number of sub-clusters expected (deck boards)

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