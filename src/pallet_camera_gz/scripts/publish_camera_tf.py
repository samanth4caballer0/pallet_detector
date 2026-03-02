#!/usr/bin/env python3
"""
Publish TF frames for the camera stand model.

TF tree:
  world -> base_link                    (dynamic, from Gazebo link_states, projected to ground)
  base_link -> rgbd_camera_link         (static, camera position relative to robot base on ground)
  rgbd_camera_link -> rgbd_camera_optical_frame  (handled by static_transform_publisher in launch)

When you later replace the static stand with a mobile robot, just change
what publishes world -> base_link (e.g. odometry / AMCL).
"""
import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import TransformStamped
import tf2_ros

STAND_LINK = "camera_stand::stand_link"

def main():
    rospy.init_node("camera_stand_tf_publisher")

    br = tf2_ros.TransformBroadcaster()
    static_br = tf2_ros.StaticTransformBroadcaster()

    # ── Static TF: base_link -> camera_link ──
    # Camera position relative to base_link (at ground level, z=0).
    # stand_link is at z=0.25, sensor pose in SDF is (0.25, 0, 0.30) relative to stand_link,
    # so absolute camera height = 0.25 + 0.30 = 0.55m above ground.
    import tf.transformations as tft
    cam_x, cam_y, cam_z = 0.25, 0.0, 0.55   # relative to base_link (ground)
    cam_roll, cam_pitch, cam_yaw = 0.0, 0.0, 0.0
    q = tft.quaternion_from_euler(cam_roll, cam_pitch, cam_yaw)

    st = TransformStamped()
    st.header.stamp = rospy.Time.now()
    st.header.frame_id = "base_link"
    st.child_frame_id = "rgbd_camera_link"
    st.transform.translation.x = cam_x
    st.transform.translation.y = cam_y
    st.transform.translation.z = cam_z
    st.transform.rotation.x = q[0]
    st.transform.rotation.y = q[1]
    st.transform.rotation.z = q[2]
    st.transform.rotation.w = q[3]
    static_br.sendTransform(st)

    # ── Dynamic TF: world -> base_link (from Gazebo) ──
    # Gazebo reports stand_link at z=0.25 (box center).
    # We project to ground level so base_link sits at z=0.
    STAND_LINK_Z_OFFSET = 0.25  # half the 0.5m stand height

    state = {"idx": None, "pose": None, "rot": None}

    def cb(msg: LinkStates):
        if state["idx"] is None:
            try:
                state["idx"] = msg.name.index(STAND_LINK)
            except ValueError:
                rospy.logwarn_throttle(5.0, f"Link '{STAND_LINK}' not found in /gazebo/link_states")
                return

        i = state["idx"]
        state["pose"] = msg.pose[i].position
        state["rot"]  = msg.pose[i].orientation

    def pub_timer(event):
        if state["pose"] is None:
            return
        p = state["pose"]
        o = state["rot"]

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "base_link"
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z - STAND_LINK_Z_OFFSET  # project to ground
        t.transform.rotation = o
        br.sendTransform(t)

    rospy.Subscriber("/gazebo/link_states", LinkStates, cb, queue_size=1)
    rospy.Timer(rospy.Duration(1.0/30.0), pub_timer)
    rospy.spin()

if __name__ == "__main__":
    main()