#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/centroid.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

class PalletDetector
{
public:
  using PointT = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<PointT>;

  PalletDetector(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh), pnh_(pnh), tf_listener_(tf_buffer_)
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
    pnh_.param("voxel_leaf", voxel_leaf_, 0.02);
    // Plane removal
    pnh_.param("plane_dist_thresh", plane_dist_thresh_, 0.02);
    pnh_.param("plane_max_iters", plane_max_iters_, 100);

    // Clustering
    pnh_.param("cluster_tolerance", cluster_tolerance_, 0.05);
    pnh_.param("cluster_min_size", cluster_min_size_, 500);
    pnh_.param("cluster_max_size", cluster_max_size_, 200000);

    // Pallet expected dims (meters) — width and height are used for the post-OBB gate
    pnh_.param("pallet_width", pallet_W_, 0.8);
    pnh_.param("pallet_height", pallet_H_, 0.145);
    pnh_.param("tol_width", tol_W_, 0.10);
    pnh_.param("tol_height", tol_H_, 0.06);

    // Template matching params
    pnh_.param("tpl_cell_size", tpl_cell_size_, 0.02);             // 2D grid resolution (m/cell)
    pnh_.param("chi_threshold", chi_threshold_, 0.40);             // minimum chi score to accept cluster
    pnh_.param("tpl_stringer_width", tpl_stringer_width_, 0.10);   // EUR/EPAL stringer width (m)
    pnh_.param("tpl_pocket_width", tpl_pocket_width_, 0.25);       // fork-pocket width (m), used only for info
    pnh_.param("tpl_top_deck_height", tpl_top_deck_height_, 0.04); // top deck thickness (m)
    pnh_.param("tpl_stringer_height", tpl_stringer_height_, 0.10); // lower stringer zone height (m)

    buildPalletTemplate();

    sub_ = nh_.subscribe(cloud_topic_, 1, &PalletDetector::cloudCb, this);

    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pallet_pose", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("pallet_marker", 1);
    roi_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_roi", 1);
    no_plane_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_no_plane", 1);

    ROS_INFO_STREAM("pallet_detector: subscribing to " << cloud_topic_);
    ROS_INFO_STREAM("pallet_detector: template " << template_cols_ << "x" << template_rows_
                                                 << " cells  mu=" << template_mu_);
  }

private: // MAIN CALLBACK
  void cloudCb(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty())
      return;

    const std::string frame = output_frame_.empty() ? msg->header.frame_id : output_frame_;

    // Log camera pose in world frame every frame
    try
    {
      geometry_msgs::TransformStamped cam_tf =
          tf_buffer_.lookupTransform("world", frame, ros::Time(0));
      const auto &t = cam_tf.transform.translation;
      const auto &r = cam_tf.transform.rotation;
      const double cam_yaw = std::atan2(2.0 * (r.w * r.z + r.x * r.y),
                                        1.0 - 2.0 * (r.y * r.y + r.z * r.z)) * 180.0 / M_PI;
      ROS_INFO_STREAM("CAMERA WORLD: pos=(" << t.x << ", " << t.y << ", " << t.z << ")"
                                            << " yaw=" << cam_yaw << " deg");
    }
    catch (const tf2::TransformException &ex)
    {
      ROS_WARN_STREAM_THROTTLE(2.0, "Camera TF lookup failed: " << ex.what());
    }

    // 1) ROI crop box
    CloudT::Ptr roi = cropBoxROI(cloud);

    if (roi_pub_.getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2 roi_msg;
      pcl::toROSMsg(*roi, roi_msg);
      roi_msg.header = msg->header;
      roi_pub_.publish(roi_msg);
    }

    // 2) Voxel downsample
    CloudT::Ptr ds(new CloudT);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(roi);
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.filter(*ds);
    if (ds->size() < 100)
      return;

    // 3) Plane segmentation (remove floor / dominant horizontal surface)
    CloudT::Ptr no_plane(new CloudT);
    removeDominantPlane(ds, no_plane);
    if (no_plane->size() < 200)
      return;

    if (no_plane_pub_.getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2 np_msg;
      pcl::toROSMsg(*no_plane, np_msg);
      np_msg.header = msg->header;
      no_plane_pub_.publish(np_msg);
    }

    // 5) Euclidean clustering
    std::vector<pcl::PointIndices> cluster_indices;
    euclideanClusters(no_plane, cluster_indices);

    ROS_WARN_STREAM("cloud=" << cloud->size()
                             << " roi=" << roi->size()
                             << " ds=" << ds->size()
                             << " no_plane=" << no_plane->size()
                             << " clusters=" << cluster_indices.size());

    // 6) For each cluster: front-face template match → back-project → OBB → 2-dim gate.
    bool candidate_found = false;
    double best_chi = -FLT_MAX;
    geometry_msgs::PoseStamped best_pose;
    visualization_msgs::Marker best_marker;

    // Height-crop window derived from floor level.
    // Camera optical frame: Y=down, so floor = large Y, pallet top = floor_y - pallet_H.
    const bool have_floor = !std::isnan(floor_y_);
    const float crop_y_hi = have_floor ? floor_y_ + static_cast<float>(tol_H_) : FLT_MAX;
    const float crop_y_lo = have_floor ? floor_y_ - static_cast<float>(pallet_H_) - static_cast<float>(tol_H_) : -FLT_MAX;
    ROS_WARN_STREAM("floor_y=" << floor_y_
                               << " crop_y=[" << crop_y_lo << ", " << crop_y_hi << "]");

    int cid = 0;
    for (const auto &idx : cluster_indices)
    {
      CloudT::Ptr cluster(new CloudT);
      cluster->reserve(idx.indices.size());
      for (int i : idx.indices)
        cluster->push_back((*no_plane)[i]);

      // 6a) Height crop: keep only points in the pallet height band above the floor.
      // This isolates the pallet structure from anything stacked on top of it.
      CloudT::Ptr face_pts(new CloudT);
      face_pts->reserve(cluster->size());
      for (const auto &pt : *cluster)
        if (pt.y >= crop_y_lo && pt.y <= crop_y_hi)
          face_pts->push_back(pt);

      ROS_WARN_STREAM("  C" << cid << " cluster=" << cluster->size()
                            << " height_crop=" << face_pts->size());

      if (static_cast<int>(face_pts->size()) < 50)
      {
        ROS_WARN_STREAM("  C" << cid << " REJECT: too few points after height crop");
        ++cid;
        continue;
      }

      // 6b) Project height-cropped cloud onto 2D grid: camera X = width cols, camera Y = height rows.
      // Z is collapsed. Side-face points land in different column positions from the front face,
      // so the sliding template will score high only when it aligns with the fork-pocket pattern.
      const Eigen::Vector3f face_origin(0.0f, 0.0f, 0.0f);
      const Eigen::Vector3f axis_u(1.0f, 0.0f, 0.0f);
      const Eigen::Vector3f axis_v(0.0f, 1.0f, 0.0f);
      std::vector<uint8_t> grid;
      int gcols = 0, grows = 0;
      float u_min = 0.0f, v_min = 0.0f;
      projectToFaceGrid(face_pts, face_origin, axis_u, axis_v, grid, gcols, grows, u_min, v_min);

      // Print observed grid as ASCII art for template comparison/tuning
      {
        ROS_WARN_STREAM("  C" << cid << " OBSERVED GRID (" << gcols << "x" << grows << "):");
        for (int r = 0; r < grows; ++r)
        {
          std::string row_str;
          for (int c = 0; c < gcols; ++c)
            row_str += grid[r * gcols + c] ? '#' : '.';
          ROS_WARN_STREAM("    row " << r << ": " << row_str);
        }
        ROS_WARN_STREAM("  C" << cid << " TEMPLATE (" << template_cols_ << "x" << template_rows_ << "):");
        for (int r = 0; r < template_rows_; ++r)
        {
          std::string row_str;
          for (int c = 0; c < template_cols_; ++c)
            row_str += template_grid_[r * template_cols_ + c] ? '#' : '.';
          ROS_WARN_STREAM("    row " << r << ": " << row_str);
        }
      }

      ROS_INFO_STREAM("  C" << cid << " grid=" << gcols << "x" << grows
                            << " need=" << template_cols_ << "x" << template_rows_);
      if (gcols < template_cols_ || grows < template_rows_)
      {
        ROS_WARN_STREAM("  C" << cid << " REJECT: grid too small for template");
        ++cid;
        continue;
      }

      // 6c) Slide template → best chi score and matched window position
      int best_co = 0, best_ro = 0;
      const double chi = slideTemplate(grid, gcols, grows, best_co, best_ro);
      ROS_INFO_STREAM("  C" << cid << " chi=" << chi << " threshold=" << chi_threshold_
                            << "  best_co=" << best_co << " best_ro=" << best_ro
                            << "  u_min=" << u_min << " v_min=" << v_min);
      if (chi < chi_threshold_)
      {
        ROS_WARN_STREAM("  C" << cid << " REJECT: chi too low");
        ++cid;
        continue;
      }

      // 6d) Width gate on matched window.
      // The template locked onto a 40-column window. That window width = pallet_W_ by construction,
      // so we verify the actual point span inside it matches expected width (sanity check).
      const float cs = static_cast<float>(tpl_cell_size_);
      const float match_x_lo = u_min + static_cast<float>(best_co) * cs;
      const float match_x_hi = u_min + static_cast<float>(best_co + template_cols_) * cs;
      const float match_y_lo = v_min + static_cast<float>(best_ro) * cs;
      const float match_y_hi = v_min + static_cast<float>(best_ro + template_rows_) * cs;
      const float matched_W = match_x_hi - match_x_lo;

      CloudT::Ptr matched_pts(new CloudT);
      matched_pts->reserve(face_pts->size());
      for (const auto &pt : *face_pts)
        if (pt.x >= match_x_lo && pt.x <= match_x_hi &&
            pt.y >= match_y_lo && pt.y <= match_y_hi)
          matched_pts->push_back(pt);

      ROS_INFO_STREAM("  C" << cid << " matched_W=" << matched_W
                            << " matched_pts=" << matched_pts->size());
      if (std::abs(matched_W - static_cast<float>(pallet_W_)) > static_cast<float>(tol_W_))
      {
        ROS_WARN_STREAM("  C" << cid << " REJECT: width gate matched_W=" << matched_W);
        ++cid;
        continue;
      }
      if (static_cast<int>(matched_pts->size()) < 30)
      {
        ROS_WARN_STREAM("  C" << cid << " REJECT: too few matched points");
        ++cid;
        continue;
      }

      // 6e) Pose: RANSAC front-face plane → normal (yaw) + inlier centroid (depth).
      //     RANSAC runs only on the stringer zone (bottom portion of height window, below
      //     the top deck) so that a box sitting on the pallet cannot contribute points.
      //     Fallback to PCA yaw + centroid Z if RANSAC yields too few inliers.
      Eigen::Vector4f c4;
      pcl::compute3DCentroid(*matched_pts, c4);

      // Stringer zone: lower tpl_stringer_height_ of the matched window (Y=down in camera)
      const float stringer_y_lo = match_y_lo + static_cast<float>(tpl_top_deck_height_);
      const float stringer_y_hi = match_y_hi;  // bottom of window = stringers

      CloudT::Ptr stringer_pts(new CloudT);
      stringer_pts->reserve(matched_pts->size());
      for (const auto &pt : *matched_pts)
        if (pt.y >= stringer_y_lo && pt.y <= stringer_y_hi)
          stringer_pts->push_back(pt);

      ROS_INFO_STREAM("  Stringer zone pts=" << stringer_pts->size()
                      << " y=[" << stringer_y_lo << "," << stringer_y_hi << "]");

      const float face_pos_y = 0.5f * (match_y_lo + match_y_hi);
      float face_pos_x = c4.x(); // fallback
      float face_pos_z = c4.z(); // fallback
      float yaw = 0.f;

      {
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
        // seg.setEpsAngle(0.44f);   // ~25 degrees
        seg.setEpsAngle(0.785f); // ~45 degrees — rejects near-horizontal planes (~67°), accepts front face up to 45° camera yaw
        seg.setDistanceThreshold(0.02f);
        seg.setMaxIterations(100);
        seg.setInputCloud(stringer_pts);

        pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        seg.segment(*plane_inliers, *coeff);

        if (static_cast<int>(plane_inliers->indices.size()) >= 30 &&
            coeff->values.size() == 4)
        {
          Eigen::Vector3f n(coeff->values[0], coeff->values[1], coeff->values[2]);
          if (n.z() > 0.f)
            n = -n; // ensure normal points toward camera (−Z)

          // Reject non-vertical planes (box top, floor): front face must have |ny| < sin(30°)
          if (std::abs(n.y()) > 0.5f)
          {
            ROS_WARN_STREAM("  Front-face RANSAC found non-vertical plane (ny=" << n.y()
                                                                                << "), falling back to PCA+centroid");
            goto ransac_fallback;
          }

          // yaw: pal_z convention → face normal = (sin(yaw), 0, -cos(yaw)) → yaw = atan2(nx, -nz)
          yaw = std::atan2(n.x(), -n.z());

          // Temporary pal_x needed for ux projection (same convention as post-block computation)
          Eigen::Vector3f pal_x_tmp(std::cos(yaw), 0.f, std::sin(yaw));
          if (pal_x_tmp.x() < 0.f)
            pal_x_tmp = -pal_x_tmp;

          // Z: inlier centroid (front-face points, side-face excluded by plane constraint)
          // X: ux-midpoint — project inliers onto pal_x (face-width direction), take geometric
          //    midpoint of [ux_min, ux_max], then convert back to camera X.
          //    This is density-independent (unequal stringer coverage doesn't bias it).
          float x_sum = 0.f, z_sum = 0.f;
          float ux_min_i = FLT_MAX, ux_max_i = -FLT_MAX;
          const int n_inliers = static_cast<int>(plane_inliers->indices.size());
          for (int idx : plane_inliers->indices)
          {
            const auto &pt = (*stringer_pts)[idx];
            x_sum += pt.x;
            z_sum += pt.z;
            const float ux = pt.x * pal_x_tmp.x() + pt.z * pal_x_tmp.z();
            ux_min_i = std::min(ux_min_i, ux);
            ux_max_i = std::max(ux_max_i, ux);
          }
          const float ix = x_sum / n_inliers;
          const float iz = z_sum / n_inliers;
          face_pos_z = iz;

          // Shift centroid laterally to the ux midpoint
          const float ix_ux = ix * pal_x_tmp.x() + iz * pal_x_tmp.z();
          const float ux_mid_i = 0.5f * (ux_min_i + ux_max_i);
          face_pos_x = ix + (ux_mid_i - ix_ux) * pal_x_tmp.x();

          ROS_INFO_STREAM("  Front-face plane: " << n_inliers
                                                 << " inliers n=(" << n.x() << "," << n.y() << "," << n.z()
                                                 << ") yaw=" << yaw * 180.f / M_PI << "deg"
                                                 << " face_x=" << face_pos_x << " (centroid_x=" << ix << ")"
                                                 << " face_z=" << face_pos_z);
        }
        else
        {
        ransac_fallback:
          ROS_WARN_STREAM("  Front-face RANSAC failed (" << plane_inliers->indices.size()
                                                         << " inliers), falling back to PCA+centroid");
          const float xm = c4.x(), zm = c4.z();
          float cxx = 0.f, cxz = 0.f, czz = 0.f;
          for (const auto &pt : *matched_pts)
          {
            const float dx = pt.x - xm, dz = pt.z - zm;
            cxx += dx * dx;
            cxz += dx * dz;
            czz += dz * dz;
          }
          yaw = 0.5f * std::atan2(2.f * cxz, cxx - czz);
          // face_pos_x and face_pos_z remain as c4 centroid fallback
        }
      }

      Eigen::Vector3f pal_x(std::cos(yaw), 0.f, std::sin(yaw));
      if (pal_x.x() < 0.f)
        pal_x = -pal_x;
      Eigen::Vector3f pal_z(-pal_x.z(), 0.f, pal_x.x());
      if (pal_z.z() > 0.f)
        pal_z = -pal_z;
      const Eigen::Vector3f pal_y = pal_z.cross(pal_x);

      Eigen::Matrix3f obb_rot;
      obb_rot.col(0) = pal_x;
      obb_rot.col(1) = pal_y;
      obb_rot.col(2) = pal_z;

      const Eigen::Vector3f obb_pos(face_pos_x, face_pos_y, face_pos_z);
      const Eigen::Vector3f dims(matched_W, match_y_hi - match_y_lo, 0.05f);

      // Debug: full pose breakdown — compare against known pallet ground truth
      ROS_INFO_STREAM("  POSE cam_frame: x=" << face_pos_x
                                             << " (c4x=" << c4.x() << " win_mid=" << 0.5f * (match_x_lo + match_x_hi) << ")"
                                             << "  z=" << face_pos_z << " (c4z=" << c4.z() << ")"
                                             << "  y=" << face_pos_y
                                             << "  yaw=" << yaw * 180.f / M_PI << "deg");

      ROS_INFO_STREAM_THROTTLE(1.0, "Cluster: chi=" << chi
                                                    << " dims=(" << dims.x() << ", " << dims.y() << ", " << dims.z() << ")"
                                                    << " pts=" << cluster->size()
                                                    << " centroid=(" << obb_pos.x() << ", " << obb_pos.y() << ", " << obb_pos.z() << ")");

      if (chi > best_chi)
      {
        best_chi = chi;
        candidate_found = true;

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

        best_marker = makeOBBMarker(msg->header, frame, obb_pos, obb_rot, dims);
      }
      ++cid;
    }

    // 7) Publish best detection or warn.
    if (candidate_found)
    {
      const double qx = best_pose.pose.orientation.x;
      const double qy = best_pose.pose.orientation.y;
      const double qz = best_pose.pose.orientation.z;
      const double qw = best_pose.pose.orientation.w;
      // Extract yaw as angle of pal_x in camera XZ plane.
      // Standard atan2(qw*qz) formula extracts world-Z rotation and gives 0 here because
      // the rotation is a 180° flip about a tilted axis (pal_y points down).
      // Instead: R[2][0]=2*(qx*qz-qy*qw) = pal_x.z, R[0][0]=1-2*(qy²+qz²) = pal_x.x
      const double yaw_deg = std::atan2(2.0 * (qx * qz - qy * qw),
                                        1.0 - 2.0 * (qy * qy + qz * qz)) *
                             180.0 / M_PI;

      ROS_INFO_STREAM("PALLET DETECTED: chi=" << best_chi
                                                            << " pos=(" << best_pose.pose.position.x
                                                            << ", " << best_pose.pose.position.y
                                                            << ", " << best_pose.pose.position.z << ")"
                                                            << " yaw=" << yaw_deg << " deg");
      pose_pub_.publish(best_pose);
      marker_pub_.publish(best_marker);

      // Log world-frame pose for ground-truth comparison
      try
      {
        geometry_msgs::PoseStamped world_pose;
        tf_buffer_.transform(best_pose, world_pose, "world", ros::Duration(0.1));
        const double wx = world_pose.pose.position.x;
        const double wy = world_pose.pose.position.y;
        const double wz = world_pose.pose.position.z;
        const double wqx = world_pose.pose.orientation.x;
        const double wqy = world_pose.pose.orientation.y;
        const double wqz = world_pose.pose.orientation.z;
        const double wqw = world_pose.pose.orientation.w;
        // Approach yaw: angle of -pal_z (forklift approach direction) in world XY.
        // R[0][2]=2*(qx*qz-qy*qw)=pal_z.x, R[1][2]=2*(qy*qz+qx*qw)=pal_z.y → negate for approach dir.
        // Reads 0° when pallet faces world +X with no rotation.
        const double wyaw_deg = std::atan2(-(2.0 * (wqy * wqz + wqx * wqw)),
                                           -(2.0 * (wqx * wqz - wqy * wqw))) * 180.0 / M_PI;
        ROS_INFO_STREAM_THROTTLE(1.0, "PALLET WORLD: pos=("
                                     << wx << ", " << wy << ", " << wz << ")"
                                     << " yaw=" << wyaw_deg << " deg");
      }
      catch (const tf2::TransformException &ex)
      {
        ROS_WARN_STREAM_THROTTLE(2.0, "TF world transform failed: " << ex.what());
      }
    }
    else
    {
      ROS_WARN_STREAM_THROTTLE(2.0, "No pallet detected. clusters=" << cluster_indices.size());
    }
  }

  // ---------------------------------------------------------------------------
  // Processing helpers (unchanged pipeline steps)
  // ---------------------------------------------------------------------------

  // Axis convention (camera optical frame): x=right, y=down, z=forward (depth)
  CloudT::Ptr cropBoxROI(const CloudT::Ptr &in)
  {
    CloudT::Ptr out(new CloudT);
    pcl::CropBox<PointT> crop;
    crop.setInputCloud(in);
    crop.setMin(Eigen::Vector4f(x_min_, y_min_, z_min_, 1.0f));
    crop.setMax(Eigen::Vector4f(x_max_, y_max_, z_max_, 1.0f));
    crop.filter(*out);
    return out;
  }

  void removeDominantPlane(const CloudT::Ptr &in, CloudT::Ptr &out)
  {
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(plane_max_iters_);
    seg.setDistanceThreshold(plane_dist_thresh_);
    seg.setInputCloud(in);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.empty())
    {
      out = in;
      floor_y_ = std::numeric_limits<float>::quiet_NaN();
      return;
    }

    // Save floor Y centroid (camera optical: Y=down, so floor = large Y)
    float y_sum = 0.0f;
    for (int i : inliers->indices)
      y_sum += (*in)[i].y;
    floor_y_ = y_sum / static_cast<float>(inliers->indices.size());

    pcl::ExtractIndices<PointT> ex;
    ex.setInputCloud(in);
    ex.setIndices(inliers);
    ex.setNegative(true); // keep NON-plane points
    ex.filter(*out);
  }

  void euclideanClusters(const CloudT::Ptr &in, std::vector<pcl::PointIndices> &clusters)
  {
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(in);

    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(cluster_min_size_);
    ec.setMaxClusterSize(cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in);
    ec.extract(clusters);
  }

  // ---------------------------------------------------------------------------
  // Template matching pipeline
  // ---------------------------------------------------------------------------

  // Build a binary EUR/EPAL end-face template from physical block geometry (paper approach).
  
  // Each solid region on the pallet face is defined as a rectangular block with a position
  // and size in metres, derived from the model SDF (pallet_realistic/model.sdf):
  
  //   Top deck  : 0.80m wide × tpl_top_deck_height tall  (full width, at top)
  //   L stringer: tpl_stringer_width wide, left edge x=0
  //   C stringer: tpl_stringer_width wide, left edge x = pallet_W/2 - stringer_W/2  (= 0.35m)
  //   R stringer: tpl_stringer_width wide, right edge = pallet_W  (left edge = 0.70m)
  
  // Fork-pocket voids fall between the stringers automatically (no explicit void blocks).
  // Row 0 = top of pallet face (smallest camera Y). Col 0 = left edge.
  
  void buildPalletTemplate()
  {
    const float cs = static_cast<float>(tpl_cell_size_);
    const float dh = static_cast<float>(tpl_top_deck_height_);
    const float sh = static_cast<float>(tpl_stringer_height_);
    const float sw = static_cast<float>(tpl_stringer_width_);
    const float pw = static_cast<float>(pallet_W_);

    template_cols_ = std::max(1, static_cast<int>(std::round(pw / cs)));
    template_rows_ = std::max(1, static_cast<int>(std::round((dh + sh) / cs)));
    template_grid_.assign(template_cols_ * template_rows_, 0);

    // Physical solid blocks: {x0, y0, width, height} in metres.
    // x0=0 is the left edge; y0=0 is the top of the face.
    struct Block
    {
      float x0, y0, w, h;
    };
    const Block blocks[] = {
        // Top deck — full width, top dh metres
        {0.f, 0.f, pw, dh},
        // Left stringer — from SDF: collision at y=-0.35, size y=0.10 → 0-based x=0..0.10
        {0.f, dh, sw, sh},
        // Centre stringer — from SDF: collision at y=0, size y=0.10 → 0-based x=0.35..0.45
        {pw / 2.f - sw / 2.f, dh, sw, sh},
        // Right stringer — from SDF: collision at y=+0.35, size y=0.10 → 0-based x=0.70..0.80
        {pw - sw, dh, sw, sh},
    };

    for (const auto &b : blocks)
    {
      // floor for start so blocks anchored to left/top always land on the right cell;
      // start + round(width) for end so block width in cells is exact.
      const int c0 = static_cast<int>(std::floor(b.x0 / cs));
      const int c1 = std::min(template_cols_, c0 + static_cast<int>(std::round(b.w / cs)));
      const int r0 = static_cast<int>(std::floor(b.y0 / cs));
      const int r1 = std::min(template_rows_, r0 + static_cast<int>(std::round(b.h / cs)));
      for (int r = r0; r < r1; ++r)
        for (int c = c0; c < c1; ++c)
          template_grid_[r * template_cols_ + c] = 1;
    }

    // Pre-compute mean occupancy μ (used in χ formula)
    int ones = 0;
    for (uint8_t v : template_grid_)
      ones += v;
    template_mu_ = static_cast<double>(ones) / static_cast<double>(template_grid_.size());

    // Print template as ASCII art for visual verification at startup
    ROS_INFO_STREAM("Pallet template (" << template_cols_ << " cols x " << template_rows_
                                        << " rows, " << tpl_cell_size_ * 100 << "cm/cell, mu=" << template_mu_ << "):");
    for (int r = 0; r < template_rows_; ++r)
    {
      std::string row_str;
      for (int c = 0; c < template_cols_; ++c)
        row_str += template_grid_[r * template_cols_ + c] ? '#' : '.';
      ROS_INFO_STREAM("  row " << r << ": " << row_str);
    }
  }

  // Rasterize front-slice points onto a 2D binary occupancy grid in (u, v) face coordinates.
  // u = dot(p - face_origin, axis_u)  [horizontal, columns]
  // v = dot(p - face_origin, axis_v)  [vertical, rows; v=0 is topmost point]
  void projectToFaceGrid(const CloudT::Ptr &slice,
                         const Eigen::Vector3f &face_origin,
                         const Eigen::Vector3f &axis_u,
                         const Eigen::Vector3f &axis_v,
                         std::vector<uint8_t> &grid,
                         int &gcols, int &grows,
                         float &u_min, float &v_min) const
  {
    std::vector<float> us, vs;
    us.reserve(slice->size());
    vs.reserve(slice->size());

    for (const auto &pt : *slice)
    {
      const Eigen::Vector3f d(pt.x - face_origin.x(),
                              pt.y - face_origin.y(),
                              pt.z - face_origin.z());
      us.push_back(d.dot(axis_u));
      vs.push_back(d.dot(axis_v));
    }

    u_min = *std::min_element(us.begin(), us.end());
    v_min = *std::min_element(vs.begin(), vs.end());
    const float u_max = *std::max_element(us.begin(), us.end());
    const float v_max = *std::max_element(vs.begin(), vs.end());

    const float cs = static_cast<float>(tpl_cell_size_);
    gcols = std::max(1, (int)std::ceil((u_max - u_min) / cs));
    grows = std::max(1, (int)std::ceil((v_max - v_min) / cs));

    grid.assign(gcols * grows, 0);
    for (size_t i = 0; i < us.size(); ++i)
    {
      const int c = std::min(gcols - 1, (int)((us[i] - u_min) / cs));
      const int r = std::min(grows - 1, (int)((vs[i] - v_min) / cs));
      grid[r * gcols + c] = 1;
    }
  }

  // Slide the pre-built pallet template over the face grid.
  // Scoring: χ = (θ − μ) / (1 − μ)  where θ = overall cell-agreement rate (TP+TN / total).
  //   χ = 0  → solid box (all occupied → only TPs, no TNs)
  //   χ = 1  → perfect structural match (stringers + void pockets)
  double slideTemplate(const std::vector<uint8_t> &grid,
                       int gcols, int grows,
                       int &best_co, int &best_ro) const
  {
    best_co = 0;
    best_ro = 0;
    double best_chi = -static_cast<double>(FLT_MAX);
    const int total = template_cols_ * template_rows_;

    for (int ro = 0; ro <= grows - template_rows_; ++ro)
    {
      for (int co = 0; co <= gcols - template_cols_; ++co)
      {
        int agree = 0;
        for (int tr = 0; tr < template_rows_; ++tr)
        {
          const int grid_base = (ro + tr) * gcols + co;
          const int tpl_base = tr * template_cols_;
          for (int tc = 0; tc < template_cols_; ++tc)
            if (template_grid_[tpl_base + tc] == grid[grid_base + tc])
              ++agree;
        }
        const double theta = static_cast<double>(agree) / total;
        const double chi = (theta - template_mu_) / (1.0 - template_mu_);
        if (chi > best_chi)
        {
          best_chi = chi;
          best_co = co;
          best_ro = ro;
        }
      }
    }
    return best_chi;
  }

  // ---------------------------------------------------------------------------
  // OBB: PCA + 2D convex hull + minimum-area bounding rectangle (unchanged)
  // ---------------------------------------------------------------------------
  bool computeOBB(const CloudT::Ptr &cluster,
                  Eigen::Vector3f &obb_pos,
                  Eigen::Matrix3f &obb_rot,
                  Eigen::Vector3f &obb_min,
                  Eigen::Vector3f &obb_max)
  {
    if (cluster->size() < 50)
      return false;

    // Step 1: PCA — smallest eigenvalue axis = thin direction (slab normal / pallet height)
    Eigen::Vector4f centroid4;
    pcl::compute3DCentroid(*cluster, centroid4);
    Eigen::Vector3f centroid = centroid4.head<3>();

    Eigen::Matrix3f cov;
    pcl::computeCovarianceMatrixNormalized(*cluster, centroid4, cov);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    Eigen::Matrix3f evecs = solver.eigenvectors(); // columns sorted by ascending eigenvalue

    Eigen::Vector3f thin_axis = evecs.col(0); // least variance
    Eigen::Vector3f u_axis = evecs.col(1);
    Eigen::Vector3f v_axis = evecs.col(2);

    // Step 2: Project onto the top-face (u, v) plane; record thin-axis coords
    std::vector<Eigen::Vector2f> pts2d(cluster->size());
    std::vector<float> thin_vals(cluster->size());

    for (size_t i = 0; i < cluster->size(); ++i)
    {
      const Eigen::Vector3f p((*cluster)[i].x, (*cluster)[i].y, (*cluster)[i].z);
      const Eigen::Vector3f d = p - centroid;
      pts2d[i] = Eigen::Vector2f(d.dot(u_axis), d.dot(v_axis));
      thin_vals[i] = d.dot(thin_axis);
    }

    // Step 3: 2D convex hull (Andrew's monotone chain)
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
    for (size_t i : order)
    {
      while (hull.size() >= 2 && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts2d[i]) <= 0.0f)
        hull.pop_back();
      hull.push_back(pts2d[i]);
    }
    const size_t lower_sz = hull.size() + 1;
    for (auto it = order.rbegin(); it != order.rend(); ++it)
    {
      while (hull.size() >= lower_sz && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts2d[*it]) <= 0.0f)
        hull.pop_back();
      hull.push_back(pts2d[*it]);
    }
    hull.pop_back();

    if (hull.size() < 3)
      return false;

    // Step 4: Minimum-area bounding rectangle via rotating calipers
    float best_area = FLT_MAX;
    float best_angle = 0.0f;
    Eigen::Vector2f best_rmin, best_rmax;

    for (size_t i = 0; i < hull.size(); ++i)
    {
      const size_t j = (i + 1) % hull.size();
      const float angle = std::atan2(hull[j].y() - hull[i].y(), hull[j].x() - hull[i].x());
      const float ca = std::cos(angle), sa = std::sin(angle);

      Eigen::Vector2f mn(FLT_MAX, FLT_MAX);
      Eigen::Vector2f mx(-FLT_MAX, -FLT_MAX);
      for (const auto &p : hull)
      {
        const float rx = ca * p.x() + sa * p.y();
        const float ry = -sa * p.x() + ca * p.y();
        mn.x() = std::min(mn.x(), rx);
        mn.y() = std::min(mn.y(), ry);
        mx.x() = std::max(mx.x(), rx);
        mx.y() = std::max(mx.y(), ry);
      }

      const float area = (mx.x() - mn.x()) * (mx.y() - mn.y());
      if (area < best_area)
      {
        best_area = area;
        best_angle = angle;
        best_rmin = mn;
        best_rmax = mx;
      }
    }

    // Step 5: Build OBB from the minimum-area rectangle
    const float ca = std::cos(best_angle), sa = std::sin(best_angle);

    Eigen::Vector3f new_u = ca * u_axis + sa * v_axis;
    Eigen::Vector3f new_v = -sa * u_axis + ca * v_axis;

    if (thin_axis.dot(new_u.cross(new_v)) < 0)
      thin_axis = -thin_axis;

    const Eigen::Vector2f rc = 0.5f * (best_rmin + best_rmax);
    const float cx_uv = ca * rc.x() - sa * rc.y();
    const float cy_uv = sa * rc.x() + ca * rc.y();

    const float thin_min = *std::min_element(thin_vals.begin(), thin_vals.end());
    const float thin_max = *std::max_element(thin_vals.begin(), thin_vals.end());
    const float thin_ctr = 0.5f * (thin_min + thin_max);

    obb_pos = centroid + cx_uv * u_axis + cy_uv * v_axis + thin_ctr * thin_axis;

    obb_rot.col(0) = new_u;
    obb_rot.col(1) = new_v;
    obb_rot.col(2) = thin_axis;

    const float half_u = 0.5f * (best_rmax.x() - best_rmin.x());
    const float half_v = 0.5f * (best_rmax.y() - best_rmin.y());
    const float half_t = 0.5f * (thin_max - thin_min);

    obb_min = Eigen::Vector3f(-half_u, -half_v, -half_t);
    obb_max = Eigen::Vector3f(half_u, half_v, half_t);

    return true;
  }

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

  // ---------------------------------------------------------------------------
  // Member variables
  // ---------------------------------------------------------------------------
private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_;
  ros::Publisher pose_pub_, marker_pub_, roi_pub_, no_plane_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string cloud_topic_;
  std::string output_frame_;

  // ROI
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;

  // Filtering
  double voxel_leaf_;
  // Plane removal
  double plane_dist_thresh_;
  int plane_max_iters_;
  float floor_y_ = std::numeric_limits<float>::quiet_NaN(); // floor Y in camera frame (Y=down)

  // Clustering
  double cluster_tolerance_;
  int cluster_min_size_;
  int cluster_max_size_;

  // Pallet dimensions (for 2-dim gate after OBB)
  double pallet_W_, pallet_H_;
  double tol_W_, tol_H_;

  // Template matching
  double tpl_cell_size_;
  double chi_threshold_;
  double tpl_stringer_width_, tpl_pocket_width_;
  double tpl_top_deck_height_, tpl_stringer_height_;
  std::vector<uint8_t> template_grid_;
  int template_cols_, template_rows_;
  double template_mu_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pallet_detector");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  PalletDetector node(nh, pnh);
  ros::spin();
  return 0;
}
